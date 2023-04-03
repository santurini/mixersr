import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import to_tensor, to_pil_image
import random

def pil_add_compression(pil_img: Image.Image, q: int) -> Image.Image:
    # BytesIO: just like an opened file, but in memory
    with BytesIO() as buffer:
        # do the actual compression
        pil_img.save(buffer, format='JPEG', quality=q)
        buffer.seek(0)
        with Image.open(buffer) as compressed_img:
            compressed_img.load()  # keep image in memory after exiting the `with` block
            return compressed_img
            
def torch_add_compression(in_tensor: torch.Tensor, q: int) -> torch.Tensor:
    pil_img = to_pil_image(in_tensor)
    compressed_img = pil_add_compression(pil_img, q=q)
    return to_tensor(compressed_img).type_as(in_tensor)

def torch_batch_add_compression(in_batch: torch.Tensor, q: int) -> torch.Tensor:
    return torch.stack([torch_add_compression(elem, q) for elem in in_batch])
    
class JPEGCompressor(torch.nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q_min = q[0]
        self.q_max = q[1]
        
    def forward(self, in_batch):
        q = random.randint(self.q_min, self.q_max)
        return torch_batch_add_compression(in_batch.detach().clamp(0, 1), q).type_as(in_batch)
