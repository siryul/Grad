import numpy as np
import torch
from torch.nn import functional as F

from utils.AverageMeter import AverageMeter
from utils.metric import accuracy, calibration


class AccMeter():

  def __init__(self, config) -> None:
    self.top1 = AverageMeter('Acc@1', ':6.3f')
    self.top5 = AverageMeter('Acc@5', ':6.3f')

    self.class_num = torch.zeros(config['num_classes'])
    self.correct = torch.zeros(config['num_classes'])

    self.confidence = np.array([])
    self.pred_class = np.array([])
    self.true_class = np.array([])

    self.config = config

  def update(self, output, target, is_prob=False):
    if not is_prob:
      output = torch.softmax(output, dim=1)

    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    self.top1.update(acc1[0], target.size(0))
    self.top5.update(acc5[0], target.size(0))

    _, predicted = output.max(1)
    target_one_hot = F.one_hot(target, self.config['num_classes'])
    predict_one_hot = F.one_hot(predicted, self.config['num_classes'])
    self.class_num = self.class_num + target_one_hot.sum(dim=0).to(torch.float)
    self.correct = self.correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

    confidence_part, pred_class_part = torch.max(output, dim=1)
    self.confidence = np.append(self.confidence, confidence_part.cpu().numpy())
    self.pred_class = np.append(self.pred_class, pred_class_part.cpu().numpy())
    self.true_class = np.append(self.true_class, target.cpu().numpy())

  def get_shot_acc(self):
    acc_classes = self.correct / self.class_num
    # for SVHN
    acc_classes = torch.cat([acc_classes, acc_classes[:1]])

    return acc_classes

  def get_cal(self):
    cal = calibration(self.true_class, self.pred_class, self.confidence, num_bins=15)
