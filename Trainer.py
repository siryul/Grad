from collections import defaultdict
from genericpath import exists
import shutil
import time
import numpy as np
from sklearn.base import is_classifier
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from utils.AccMeter import AccMeter
from utils.AverageMeter import AverageMeter
from utils.ProgressMeter import ProgressMeter
from utils.metric import accuracy


class Trainer:

  def __init__(self, config, model, classifier1, classifier2, train_loader, val_loader,
               balance_loader, criterion, optimizer):
    self.config = config
    self.model = model
    # obtain GradCAM back images, use cross entropy loss to give more attentions to the head class and get the context (background knowledge)
    self.classifier1 = classifier1
    # final classifier, use self defination loss function to give more weight to the target class
    self.classifier2 = classifier2
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.balance_loader = balance_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.move_to_device()
    global best_acc
    best_acc = defaultdict(float)

  def move_to_device(self):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.
                          is_available() else 'cpu')
    self.model.to(device)
    self.classifier1.to(device)
    self.classifier2.to(device)
    print("Model moved to device: ", device)

  def train(self):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(len(self.train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format)
    print("Start training...")

    self.model.train()
    self.classifier1.train()
    for epoch in range(self.config['epochs']):
      training_data_num = len(self.train_loader.dataset)
      end_steps = int(training_data_num / self.config['batch_size'])

      back_images = torch.tensor([]).to(device)
      back_masks = torch.tensor([]).to(device)

      balance_loader_iter = iter(self.balance_loader)

      end = time.time()
      for i, (images1, labels1) in enumerate(self.train_loader):
        if i >= end_steps:
          break

        # to use GradCAM back images and masks for generating new training data
        images2, labels2 = next(balance_loader_iter)
        images2, labels2 = images2[:images1.size(0)], labels2[:images1.size(0)]
        # print(f"images2.shape: {images2.shape}") # [128, 3, 32, 32]
        # print(f"labels2.shape: {labels2.shape}") # [128]

        images1, labels1 = images1.to(device), labels1.to(device)
        images2, labels2 = images2.to(device), labels2.to(device)
        # print(images[0], labels[0])

        # separate GradCAM back images and masks
        masks, logits = get_background_mask(self.model, self.classifier1, images1, labels1,
                                            self.config)
        # print(f'masks.shape: {masks.shape}')
        # print(f'logits.shape: {logits.shape}')
        # compuate the probability of the label
        prob = F.softmax(logits, dim=1)
        # filter out the samples with low probability
        fit = (prob[labels1 >= 0, labels1] >= self.config['fit_threshold'])
        # print(f'fit.shape: {fit.shape}')
        # add back images and masks which fit the threshold to the bank
        back_images = torch.cat([back_images, images1[fit]], dim=0)[-self.config['bank_size']:]
        back_masks = torch.cat([back_masks, masks[fit]], dim=0)[-self.config['bank_size']:]

        # generate new training data
        if back_images.shape[0] >= images1.shape[0] and epoch >= self.config['start_aug']:
          perm = np.random.permutation(back_images.shape[0])
          aug_images, aug_masks = back_images[perm][:images1.shape[0]], back_masks[perm][:images1.
                                                                                         shape[0]]
          lam = np.random.uniform(0, 1)

          # because only use background images (not containing the target class),
          # so only need to change the images2, not the labels2
          # get background images
          images2 = lam*aug_masks*aug_images + images2 * (1. - lam*aug_masks)

          # denormalize the images, convert range to [0,255]
          # def denormalize(tensor):
          #   for t, m, s in zip(tensor, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]):
          #     t.mul_(s).add_(m)
          #   return tensor

          # plt.imshow(denormalize(images2[0].cpu().squeeze()).permute(1, 2, 0).numpy())
          # plt.show()

          feat2 = self.model(images2)
          outputs2 = self.classifier2(feat2)
          loss2 = F.cross_entropy(outputs2, labels2)
        else:
          loss2 = 0

        # calculate self defination loss
        # with torch.no_grad():
        outputs1 = self.classifier1(self.model(images1))
        if self.config['criterion'] == 'DFL':
          loss1 = self.criterion(outputs1, labels1, epoch, self.config['epochs'])
        else:
          loss1 = self.criterion(outputs1, labels1)

        # print(f'loss1: {loss1}, loss2: {loss2}')
        self.optimizer.zero_grad()
        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()

        # calculate accuracy
        acc1, acc5 = accuracy(outputs1, labels1, topk=(1, 5))
        losses.update(loss.item(), images1.size(0))
        top1.update(acc1.item(), images1.size(0))
        top5.update(acc5.item(), images1.size(0))

        batch_time.update(time.time() - end)

        if i % 100 == 0:
          print(
            f'Epoch: {epoch+1}/{self.config["epochs"]}, Step: {i+1}/{len(self.train_loader)}, Loss: {loss.item()}, Accuracy@1: {acc1.item()}, Accuracy@5: {acc5.item()}'
          )
        end = time.time()

      # start evaluate
      is_classifier1_best, is_classifier2_best = self.evaluate()
      save_checkpoints(
        {
          'epoch': epoch + 1,
          'state_dict': self.model.state_dict(),
          'classifier1': self.classifier1.state_dict(),
          'classifier2': self.classifier2.state_dict(),
          'acc1': acc1,
          'acc5': acc5,
        }, is_classifier2_best, self.config)

  def evaluate(self):
    batch_time = AverageMeter('Time', ':6.3f')
    acc_meter = {'classifier1': AccMeter(self.config), 'classifier2': AccMeter(self.config)}
    progress = ProgressMeter(len(
      self.val_loader), [batch_time, acc_meter['classifier1'].top1, acc_meter['classifier1'].top5],
                             prefix='Eval: ')
    print("Evaluating starts...")
    self.model.eval()
    self.classifier1.eval()
    self.classifier2.eval()
    correct1 = 0
    correct2 = 0
    total = 0
    with torch.no_grad():
      end = time.time()
      for i, (images, labels) in enumerate(self.val_loader):
        images, labels = images.to(device), labels.to(device)

        feat = self.model(images)
        outputs1 = self.classifier1(feat)
        outputs2 = self.classifier2(feat)

        acc_meter['classifier1'].update(outputs1.to('cpu'), labels.to('cpu'))
        acc_meter['classifier2'].update(outputs2.to('cpu'), labels.to('cpu'))

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % 100 == 0:
        #   progress.display(i, logger)

      global best_acc
      is_classifier1_best = False
      is_classifier2_best = False

      # calculate accuracy for classifier1 & classifier2

      for name in acc_meter.keys():
        entry = acc_meter[name]

        acc1, acc5 = entry.top1.avg, entry.top5.avg
        is_best = acc1 > best_acc[name]

        if is_best:
          best_acc[name] = acc1
          if name == 'classifier1':
            is_classifier1_best = True
          elif name == 'classifier2':
            is_classifier2_best = True

        print(
          f'{name} Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}\nBest Acc@1: {best_acc[name]:.3f}'
        )

      return is_classifier1_best, is_classifier2_best


def is_best(acc):
  global best_acc
  return acc > best_acc


def save_checkpoints(state, is_best, config):
  if not os.path.exists(config['ckps']):
    os.makedirs(config['ckps'])
  file_name = config['ckps'] + '/current.pth.tar'
  torch.save(state, file_name)
  if is_best:
    shutil.copyfile(file_name, config['ckps'] + '/model_best.pth.tar')


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_background_mask_by_grad_cam(model, classifier, inputs, targets):
  # use pytorch_grad_cam to generate background and mask
  # save model status
  training_state = model.training
  # set target layer
  target_layer = [model.last_layer]
  targets_ = [ClassifierOutputTarget(t) for t in targets]

  with GradCAM(model=model, target_layer=target_layer) as cam:
    grayscale_cam = cam(input_tensor=inputs, targets=targets_)
    grayscale_cam = grayscale_cam[0, :]
    # print(f'grayscale_cam: {grayscale_cam}')
    # print(f'grayscale_cam.shape: {grayscale_cam.shape}')


feat_map_global = None
grad_map_global = None


def _hook_a(module, input, output):
  global feat_map_global
  feat_map_global[output.device.index] = output


def _hook_g(module, grad_in, grad_out):
  global grad_map_global
  grad_map_global[grad_out[0].device.index] = grad_out[0]


def get_background_mask(model, classifier, images, labels, config):
  # target layer is the last convelution layer
  target_layer = model.last_layer
  fc_layer = classifier.weight

  # print("Getting background mask...")
  # print('target_layer:', target_layer)
  # print('fc_layer:', fc_layer)

  hook_a = target_layer.register_forward_hook(_hook_a)
  hook_g = target_layer.register_full_backward_hook(_hook_g)

  training_mode = model.training
  model.eval()
  classifier.eval()

  global feat_map_global
  global grad_map_global
  feat_map_global = {}
  grad_map_global = {}

  with torch.no_grad():
    feat = model.forward_1(images)
  feat = model.forward_2(feat.detach())
  outputs = classifier(feat)
  loss = outputs[labels >= 0, labels].sum()
  model.zero_grad()
  classifier.zero_grad()
  loss.backward(retain_graph=False)

  hook_a.remove()
  hook_g.remove()

  device_id = images.device.index
  feat_map = feat_map_global[device_id]
  grad_map = grad_map_global[device_id]

  # print("feat_map.shape:", feat_map.shape) # [128, 64, 8, 8] in cifar10
  # print("grad_map.shape:", grad_map.shape) # [128, 64, 8, 8] in cifar10

  with torch.no_grad():
    weights = grad_map.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat_map).sum(dim=1, keepdim=True)
    cam = F.relu(cam, inplace=True)

  def _normalize(x):
    x.sub_(x.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
    x.div_(x.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

  _normalize(cam)

  images_h, images_w = images.shape[-2], images.shape[-1]
  # print("images_h, images_w:", images_h, images_w)
  resized_cam = F.interpolate(cam, size=(images_h, images_w), mode='bilinear', align_corners=False)
  resized_cam = resized_cam.clamp(0, 1)
  mask = (1 - resized_cam)**2

  model.train(training_mode)
  classifier.train(training_mode)

  return mask, outputs.detach()
