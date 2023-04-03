import argparse
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from pathlib import Path
import math

#@title dataloaders
class SRData(torch.utils.data.Dataset):
  def __init__(self, path, size):
    self.size = size # crop size
    self.paths = list(Path(path).glob('*')) # list of videos

  def load_img(self, path):
    return to_tensor(Image.open(path))

  def crop(self, img):
    height = img.shape[1]
    width = img.shape[2]
    y_offset = int(math.ceil((height - self.size[0]) / 2))
    x_offset = int(math.ceil((width - self.size[1]) / 2))
    return img[:, y_offset : y_offset + self.size[0], x_offset : x_offset + self.size[1]]

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    img = self.load_img(self.paths[idx])
    return self.crop(img)