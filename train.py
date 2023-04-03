import argparse
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from dataset import SRData
from modules import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger, WandbLogger
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, required=False, default=16, help='batch size')
parser.add_argument('--path', type=str, required=True, help='input folder path')
parser.add_argument('--img_size', nargs=2, type=int, required=False, default=(384, 384), metavar=('height', 'width'), help='img input size')
parser.add_argument('--dct_ks', type=int, required=False, default=8, help='DCT kernel size')
parser.add_argument('--exp_fc', type=int, required=False, default=2, help='MLP expansion factor')
parser.add_argument('--up_fc', type=int, required=False, default=2, help='Upscale Factor')
parser.add_argument('--blocks', type=int, required=False, default=4, help='Mixing blocks')
parser.add_argument('--decoder', type=str, required=False, default='PS', help='PS = pixel shuffle, T = transpose convolution')
parser.add_argument('--jpeg_q', nargs=2, type=int, required=False, default=(20, 80), metavar=('min_q', 'max_q'), help='JPEG compression quality')
parser.add_argument('--epochs', type=int, required=False, default=250, help='train epochs')
parser.add_argument('--gpu', action='store_true', help='Boolean flag to use GPU')
parser.add_argument('--lr', type=float, required=False, default=2e-4, help='learning rate')
parser.add_argument('--sched', type=str, required=False, default='cosine', help='cosine - exponential')
parser.add_argument('--ckpt_path', type=str, required=False, default=None, help='path to ckeckpoint')
parser.add_argument('--wandb_key', type=str, required=False, default='./wandb_key.txt', help='wandb key path')
parser.add_argument('--wandb_proj', type=str, required=False, default='video-super-resolution', help='wandb project')

args           = parser.parse_args()
path           = args.path
bs             = args.bs
size           = tuple(args.img_size)
patch_size     = args.dct_ks
exp            = args.exp_fc
upscale_factor = args.up_fc
blocks         = args.blocks
q              = tuple(args.jpeg_q)
epochs         = args.epochs
accelerator    = 'gpu' if args.gpu else 'cpu'
ckpt_path      = args.ckpt_path

patches_dim = int(((size[0]/upscale_factor)/patch_size)**2)
channels_dim = 3*patch_size**2
h = w = int(math.sqrt(patches_dim * patch_size ** 2) / patch_size)

if args.wandb_key is not None:
    with open(args.wandb_key, 'r') as file: 
        wandb_key = file.readline().strip()
    wandb.login(key=wandb_key)    
    logger = WandbLogger(project=args.wandb_proj)
    log_imgs = True

else:
    logger = Logger()
    log_imgs = False

decoder = DecoderIDCT(channels_dim, upscale_factor, h, w, patch_size) if args.decoder.lower()=='ps' else SuperResolver(channels_dim, h, w, patch_size)
model = MixerSR(patch_size, size, exp, upscale_factor, blocks, decoder)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-07) if args.sched=='cosine' else ExponentialLR(optimizer, gamma=0.95)
lr_monitor = pl.callbacks.LearningRateMonitor('step')
ckpt = pl.callbacks.ModelCheckpoint(dirpath = './ckpt',
                                   filename = 'ckpt', 
                                   verbose = True, 
                                   monitor = 'val_loss', 
                                   mode = 'min')

tr_dl = DataLoader(SRData(os.path.join(path,'train/'), size), batch_size=bs, shuffle=True, num_workers=0)
val_dl = DataLoader(SRData(os.path.join(path,'valid/'), size), batch_size=1, shuffle=False, num_workers=0)

pl_model = LightningMixerSR(model, size, optimizer, scheduler, upscale_factor, log_imgs, q)
trainer = pl.Trainer(callbacks=[ckpt, lr_monitor], accelerator=accelerator, max_epochs=epochs, logger=logger)
trainer.fit(pl_model, tr_dl, val_dl, ckpt_path=ckpt_path)
