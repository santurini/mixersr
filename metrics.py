import torch
import torch.nn as nn

#@title psnr
class PSNR(nn.Module):
    def __init__(self, max=1.):
       super().__init__()
       self.max = max
       self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        loss = self.mse(yhat.clamp(0, self.max), y.clamp(0, self.max))      
        return 10 * torch.log10(self.max ** 2 / (loss + 1e-8))