import argparse
import os

import torch
import yaml

from loss.DFL import DFL
from loss.focal_loss import FocalLoss
from models import resnet_cifar
from Trainer import Trainer
from dataset.cifar_lt import CIFAR10_LT, CIFAR100_LT


def parse_args():
  config_file_path = os.path.join(os.getcwd(), parser.parse_args().cfg)
  with open(config_file_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  with open('./config/data.yaml', 'r') as f:
    dataset_path = yaml.load(f, Loader=yaml.FullLoader)

  config['dataset_path'] = dataset_path['data_path'][config['dataset']]
  config['criterion'] = parser.parse_args().criterion
  return config


def main():
  # parse arguments
  config = parse_args()
  print(config)

  # train model
  if config['dataset'] == 'cifar10' or config['dataset'] == 'cifar100':
    model = getattr(resnet_cifar, config['backbone'])()
    classifier1 = getattr(resnet_cifar, 'Classifier')(feat_in=config['feat_size'],
                                                      num_classes=config['num_classes'])
    classifier2 = getattr(resnet_cifar, 'Classifier')(feat_in=config['feat_size'],
                                                      num_classes=config['num_classes'])
  else:
    raise ValueError('Unsupported dataset: {}'.format(config['dataset']))

  # load data
  if config['dataset'] == 'cifar10':
    train_loader, val_loader, balanced_loader = CIFAR10_LT(root=config['dataset_path'],
                                                           imb_factor=config['imb_factor'],
                                                           batch_size=config['batch_size'],
                                                           num_workers=config['num_workers'])

  elif config['dataset'] == 'cifar100':
    train_loader, val_loader, balanced_loader = CIFAR100_LT(root=config['dataset_path'],
                                                            imb_factor=config['imb_factor'],
                                                            batch_size=config['batch_size'],
                                                            num_workers=config['num_workers'])
  else:
    raise ValueError('Unsupported dataset: {}'.format(config['dataset']))

  if config['criterion'] == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
  elif config['criterion'] == 'FL':
    criterion = FocalLoss()
  elif config['criterion'] == 'DFL':
    criterion = DFL()
  else:
    raise ValueError('Unsupported criterion: {}'.format(config['criterion']))

  parameters = [{'params': model.parameters()}, {'params': classifier1.parameters()}]
  if config['dataset'] in ['cifar10', 'cifar100']:
    parameters.append({'params': classifier2.parameters()})

  optimizer = torch.optim.SGD(parameters,
                              lr=config['lr'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'])

  print('=' * 50, '\n', len(balanced_loader.dataset))

  trainer = Trainer(config,
                    model,
                    classifier1,
                    classifier2,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    balance_loader=balanced_loader,
                    criterion=criterion,
                    optimizer=optimizer)
  trainer.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg',
                      type=str,
                      default='config/cifar10/cifar10_lt_01.yaml',
                      help='Path to config file')
  parser.add_argument('--criterion', type=str, default='FL', help='Criterion to use')
  main()
