from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class HAMDataset(Dataset):

  def __init__(self, img_paths, labels, transform=None) -> None:
    super().__init__()
    self.img_paths = img_paths
    self.labels = labels
    self.classes = np.unique(labels).tolist()
    self.num_classes = len(self.classes)
    self.transform = transform

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, index):
    image = Image.open(self.img_paths[index])
    if self.transform:
      image = self.transform(image)
    label = self.classes.index(self.labels[index])
    return image, label


class HAM10000():

  def __init__(self, root, config) -> None:
    train_loader = HAMDataset(root['train'], )
