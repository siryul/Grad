import torch
import torch.nn as nn
import torch.nn.functional as F


class DFL(nn.Module):

  def __init__(self, alpha=0.25, gamma=2, reduction='mean') -> None:
    super(DFL, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, input: torch.Tensor, target: torch.Tensor, current_epoch: int,
              total_epoch: int) -> torch.Tensor:
    ce_loss = F.cross_entropy(input, target, reduction='none')
    pt = torch.exp(-ce_loss)
    progress = current_epoch / total_epoch
    # dfl = ce_loss + self.alpha * (1 - pt)**(self.gamma * current_epoch / total_epoch) * ce_loss
    dfl = (1-progress) * ce_loss + progress * (1 - pt)**(self.gamma) * ce_loss
    if self.reduction == 'mean':
      return dfl.mean()
    elif self.reduction == 'sum':
      return dfl.sum()
    else:
      return dfl
