# This file is used to generate the CIFAR-LT dataset.
# implementation of CIFAR-LT dataset is based on the paper "Learning imbalanced datasets with label-distribution-aware margin loss"
# https://arxiv.org/abs/2007.03321
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from dataset.sampler import ClassAwareSampler

__all__ = ['CIFAR10_LT', 'CIFAR100_LT']


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
  cls_num = 10

  def __init__(self,
               root,
               imb_type='exp',
               imb_factor=0.01,
               rand_number=0,
               train=True,
               transform=None,
               target_transform=None,
               download=False):
    super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
    np.random.seed(rand_number)
    self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
    print(f'num_cls_list: {self.img_num_list}')
    self.gen_imbalanced_data(self.img_num_list)

  def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    img_max = len(self.data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
      for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num-1.0)))
        img_num_per_cls.append(int(num))
    elif imb_type == 'step':
      for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max))
      for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max * imb_factor))
    else:
      img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

  def gen_imbalanced_data(self, img_num_per_cls):
    new_data = []
    new_targets = []
    targets_np = np.array(self.targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    self.num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
      self.num_per_cls_dict[the_class] = the_img_num
      idx = np.where(targets_np == the_class)[0]
      np.random.shuffle(idx)
      selec_idx = idx[:the_img_num]
      new_data.append(self.data[selec_idx, ...])
      new_targets.extend([
        the_class,
      ] * the_img_num)
    new_data = np.vstack(new_data)
    self.data = new_data
    self.targets = new_targets

  def get_cls_num_list(self):
    cls_num_list = []
    for i in range(self.cls_num):
      cls_num_list.append(self.num_per_cls_dict[i])
    return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
  """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
  This is a subclass of the `CIFAR10` Dataset.
  """
  base_folder = 'cifar-100-python'
  url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
  filename = "cifar-100-python.tar.gz"
  tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
  train_list = [
    ['train', '16019d7e3df5f24257cddd939b257f8d'],
  ]

  test_list = [
    ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
  ]
  meta = {
    'filename': 'meta',
    'key': 'fine_label_names',
    'md5': '7973b15100ade9c7d40fb424638fde48',
  }
  cls_num = 100


train_transform = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

val_transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def CIFAR10_LT(root, imb_type='exp', imb_factor=0.01, batch_size=128, num_workers=4):
  train_dataset = IMBALANCECIFAR10(root=root,
                                   imb_type=imb_type,
                                   imb_factor=imb_factor,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
  val_dataset = CIFAR10(root=root, train=False, transform=val_transform, download=True)
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
  val_loader = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers)
  # 平衡采样，用于训练过程中使用 Mixup 等方法
  # weight = train_dataset.img_num_list
  # weight = [1.0 / i for i in weight]
  # sample_weight = torch.tensor([weight[i] for i in train_dataset.targets]).double()
  # print(f'sample_weight: {sample_weight}')
  # balanced_sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
  balanced_sampler = ClassAwareSampler(train_dataset)
  balanced_dataloader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=balanced_sampler,
                                   num_workers=num_workers)
  return train_loader, val_loader, balanced_dataloader


def CIFAR100_LT(root, imb_type='exp', imb_factor=0.01, batch_size=128, num_workers=4):
  train_dataset = IMBALANCECIFAR100(root=root,
                                    imb_type=imb_type,
                                    imb_factor=imb_factor,
                                    train=True,
                                    transform=train_transform,
                                    download=True)
  val_dataset = CIFAR100(root=root, train=False, transform=val_transform, download=True)
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
  val_loader = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers)
  # 平衡采样，用于训练过程中使用 Mixup 等方法
  # weight = train_dataset.img_num_list
  # weight = [1.0 / i for i in weight]
  # sample_weight = torch.tensor([weight[i] for i in train_dataset.targets]).double()
  # print(f'sample_weight: {sample_weight}')
  # balanced_sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
  balanced_sampler = ClassAwareSampler(train_dataset)
  balanced_dataloader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=balanced_sampler,
                                   num_workers=num_workers)
  return train_loader, val_loader, balanced_dataloader


if __name__ == '__main__':
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = IMBALANCECIFAR100(root='./data', train=True, download=True, transform=transform)
  trainloader = iter(trainset)
  data, label = next(trainloader)
  import pdb
  pdb.set_trace()
