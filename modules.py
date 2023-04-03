import torch
import torch.nn as nn
from torchvision.transforms import Resize
import math
import numpy as np
import pytorch_lightning as pl
from einops.layers.torch import *
from imgaug.augmenters.arithmetic import JpegCompression
from utils import JPEGCompressor
from metrics import PSNR

class EncoderDCT(nn.Module):
    def __init__(self, ps=4):
        super(EncoderDCT, self).__init__()
        self.dct_conv = nn.Conv2d(3, 3*ps*ps, ps, ps, bias=False, groups=3) 
        matrix = self.generate_dct_matrix(ps)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1)        
        self.dct_conv.weight.data = torch.cat([self.weight] * 3, dim=0) 
        self.dct_conv.weight.requires_grad = False
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        
    def forward(self, x):
        dct = self.dct_conv(x)
        return self.rearrange(dct)
    
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0: 
            return result
        return result * math.sqrt(2)
    
    def generate_dct_matrix(self,ps=8):
        matrix = np.zeros((ps, ps, ps,ps))  
        for u in range(ps):
            for v in range(ps):
                for i in range(ps):
                    for j in range(ps):
                        matrix[u, v, i, j] = self.build_filter(i, u, ps) * self.build_filter(j, v, ps)   
        return matrix.reshape(-1,ps, ps)

class Mlp(nn.Module):
  def __init__(self, dim, in_dim):
    super(Mlp, self).__init__()
    self.mlp = nn.Sequential(nn.Linear(dim, in_dim),
                             nn.GELU(),
                             nn.Linear(in_dim, dim))
  def forward(self, x):
    return self.mlp(x)

class MixerBlock(nn.Module):
  def __init__(self, patches_dim, channels_dim, exp):
    super(MixerBlock, self).__init__()
    self.layer_norm = nn.LayerNorm(channels_dim)
    self.patches_mlp = Mlp(patches_dim, exp*patches_dim)
    self.channels_mlp = Mlp(channels_dim, exp*channels_dim)

  def forward(self, x):
    out = self.patches_mlp(self.layer_norm(x).mT)
    out = self.channels_mlp(self.layer_norm(out.mT + x)) + (out.mT + x)
    return out

class MlpMixer(nn.Module):
  def __init__(self, patch_size, patches_dim, channels_dim, h, w, exp, blocks):
    super(MlpMixer, self).__init__()
    self.mixer = nn.Sequential(*[MixerBlock(patches_dim, channels_dim, exp) for _ in range(blocks)])

  def forward(self, x):
    return self.mixer(x)

### MixerSRT Modules Only ###

class UpConv(nn.Module):
    def __init__(self, in_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(in_ch, 128, 1, 1, 0, bias=False),
                                nn.SELU(),
                                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                nn.SELU(),
                                nn.ConvTranspose2d( 64, 32, 4, 2, 1, bias=False),
                                nn.SELU(),
                                nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
                                nn.SELU(),
                                nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False))
    
    def forward(self, x):
        return self.conv(x)
    
class SuperResolver(nn.Module):
    def __init__(self, in_ch, h, w, ps=4):
        super().__init__()
        self.rearrange = Rearrange('b (h w) c -> b c h w', h=h, w=w)
        self.upscaler = UpConv(in_ch)
        
    def forward(self, x):
        x = self.rearrange(x)
        x = self.upscaler(x)
        return x

### MixerSRPS Modules Only ###

class DecoderIDCT(nn.Module):
    def __init__(self, channels_dim, upscale_factor, h, w, ps=4):
        super(DecoderIDCT, self).__init__()
        self.reverse_dct_conv = nn.ConvTranspose2d(3 * ps * ps, 3, ps, ps, bias=False, groups=3)
        matrix = self.generate_dct_matrix(ps)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1) 
        self.reverse_dct_conv.weight.data = torch.cat([self.weight] * 3, dim=0) 
        self.reverse_dct_conv.weight.requires_grad = False
        self.projector = nn.Linear(channels_dim, channels_dim * upscale_factor ** 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.rearrange = Rearrange('b (h w) c -> b c h w', h=h, w=w)
        
    def forward(self, x):
        x = self.rearrange(self.projector(x))
        x = self.pixel_shuffle(x)
        return self.reverse_dct_conv(x)
    
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0: 
            return result
        return result * math.sqrt(2)
    
    def generate_dct_matrix(self,ps=8):
        matrix = np.zeros((ps, ps, ps,ps))  
        for u in range(ps):
            for v in range(ps):
                for i in range(ps):
                    for j in range(ps):
                        matrix[u, v, i, j] = self.build_filter(i, u, ps) * self.build_filter(j, v, ps)   
        return matrix.reshape(-1,ps, ps)

##############################

class MixerSR(nn.Module):
    def __init__(self, patch_size, size, exp, upscale_factor, blocks, decoder):
        super().__init__()
        patches_dim = int(((size[0]/upscale_factor)/patch_size)**2)
        channels_dim = 3*patch_size**2
        h = w = int(math.sqrt(patches_dim * patch_size ** 2) / patch_size)
        self.encoder = EncoderDCT(patch_size)
        self.mlp_mixer = MlpMixer(patch_size, patches_dim, channels_dim, h, w, exp, blocks)
        self.decoder = decoder
        self.upscale = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')
    
    def forward(self, x):
        lr = x
        x = self.encoder(x)
        x = self.mlp_mixer(x)
        return self.decoder(x) + self.upscale(lr)
    
### Lightning Module ###

#@title lightning
class LightningMixerSR(pl.LightningModule):
    def __init__(self, model, size, optimizer, scheduler, upscale_factor, log_imgs, q):
        super().__init__()
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.L1Loss()
        self.metric = PSNR()
        self.log_imgs = log_imgs
        self.resize = Resize((size[0]//upscale_factor, size[1]//upscale_factor))
        self.compression = JPEGCompressor(q)
        
    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, batch_idx, stage):
        log_flag = (batch_idx % 30 == 0) & self.log_imgs
        compressed = self.compression(self.resize(batch))
        output = self.forward(compressed)
        loss = self.criterion(output, batch)
        psnr = self.metric(output, batch)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_psnr', psnr, prog_bar=True)
        if log_flag:
          self.log_images(batch, output, compressed)
        return {"loss": loss, "psnr": psnr}
    
    def shared_epoch_end(self, outputs, stage):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_psnr = torch.tensor([x["psnr"] for x in outputs]).mean()
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train") 

    def on_train_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")    

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def log_images(self, hr, out, inp):
        gt = [i for i in hr[:3]]
        pred = [i for i in out[:3]]
        inp = [i for i in inp[:3]]
        psnr = ['PSNR: ' + str(self.metric(i, j).detach().cpu().numpy().round(2)) for i, j in zip(pred, gt)]
        self.logger.log_image(key='Ground Truths', images=gt, caption=['gt']*3)
        self.logger.log_image(key='Predicted Images', images=pred, caption=psnr)
        self.logger.log_image(key='Input Images', images=inp, caption=['inp']*3)
        
