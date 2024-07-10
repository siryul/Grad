import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

  def __init__(self, alpha=0.25, gamma=2, reduction='mean') -> None:
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce_loss = F.cross_entropy(input, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
    if self.reduction == 'mean':
      return focal_loss.mean()
    elif self.reduction == 'sum':
      return focal_loss.sum()
    else:
      return focal_loss
