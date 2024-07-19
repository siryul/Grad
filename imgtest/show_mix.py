from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision import transforms


# 将归一化后的 tensor 还原，并进行图片展示
def denoralize(tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
  t = tensor.clone()
  if len(t.shape) == 4: # B * C * H * W
    t = make_grid(t.cpu())
  t = t.numpy().transpose((1, 2, 0))
  img = t*std + mean
  img = np.clip(img, 0, 1)
  return img


def show_mix(
  mixed: torch.Tensor,
  mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
  std: np.ndarray = np.array([0.229, 0.224, 0.225])
) -> None:
  mixed = denoralize(mixed, mean, std)
  plt.imshow(mixed)
  plt.show()


# 将 tensor 归一化至 0-1
def normalize(tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
  tensor = tensor.clone().float()
  transform = transforms.Normalize(mean=mean, std=std)
  return transform(tensor)
  # return tensor
  # print(tensor.dtype)
  # for t, m, s in zip(tensor, mean, std):
  #   # print(t.dtype)
  #   t.sub_(m).div_(s)
  # return tensor


if __name__ == '__main__':
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  # print(normalize(torch.arange(48).reshape(3, 4, 4)))
  a = torch.arange(48).reshape(3, 4, 4)
  print(len(a.shape))
  # normalized_a = normalize(a, mean, std)
  # print(normalized_a)
  # print('---', denoralize(normalized_a, mean, std))
