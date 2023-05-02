# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for VAEBM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
'''Code for training VAEBM'''
import time
import random
import math
import sys
import json
import glob
import argparse
import torch
from addict import Dict
import numpy as np
import os
import logging
from torch import autograd
import torch.distributed as dist
import torchvision.datasets as dset
from torch.multiprocessing import Process
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from torch import optim
import utils
from torch.utils.tensorboard import SummaryWriter

import distributions
import DCGAN_VAE_pixel as DVAE
import datasets
import torchvision
from tqdm import tqdm
from ebm_models import EBM_CelebA64, EBM_LSUN64, EBM_CIFAR32, EBM_CelebA256
from thirdparty.igebm_utils import sample_data, clip_grad


def cleanup():
    dist.destroy_process_group()
    
def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


class SampleBuffer:
    def __init__(self, num_block, max_samples, device = torch.device('cuda:0')):
        self.max_samples = max_samples
        self.num_block = num_block
        self.buffer = [[] for _ in range(num_block)]  #each group of latent variable is a list
        self.device = device

    def __len__(self):
        return len(self.buffer[0]) #len of the buffer should be the length of list for each group of latent

    def push(self, z_list): #samples is a list of torch tensor
        for i in range(self.num_block):
            zi = z_list[i]
            zi = zi.detach().to('cpu')
            for sample in zip(zi):
                self.buffer[i].append(sample[0])
                if len(self.buffer[i]) > self.max_samples:
                    self.buffer[i].pop(0)

    def get(self, n_samples):
        sample_idx = random.sample(range(len(self.buffer[0])), n_samples)
        z_list = []
        for i in range(self.num_block):
            samples = [self.buffer[i][j] for j in sample_idx]
            samples = torch.stack(samples, 0)
            samples = samples.to(self.device)
            z_list.append(samples)

        return z_list
    def save(self,fname):
        torch.save(self.buffer,fname)




def sample_buffer(buffer, z_list_exampler, batch_size=64, t = 1, p=0.95, device=torch.device('cuda:0')):
    if len(buffer) < 1:       
        
        eps_z = [torch.Tensor(batch_size, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).to(device) \
                 for zi in z_list_exampler]

        return eps_z
    

    n_replay = (np.random.rand(batch_size) < p).sum()
    
    if n_replay > 0:
    
        eps_z_replay = buffer.get(n_replay)
        eps_z_prior = [torch.Tensor(batch_size - n_replay, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).to(device)\
                for zi in z_list_exampler]

        eps_z_combine = [torch.cat([z1,z2], dim = 0) for z1,z2 in zip(eps_z_replay, eps_z_prior)]
        
        return eps_z_combine
    else:
        eps_z = [torch.Tensor(batch_size - n_replay, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).to(device) \
                for zi in z_list_exampler]

    
        return eps_z


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main(opt):
    opt.save = os.path.join(opt.save, opt.dataset, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(opt.save, scripts_to_save=glob.glob("src/*.py"))
    with open(os.path.join(opt.save, "args.json"), "w") as f:
        json.dump(opt, f, indent=4)
    
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(opt.save, "log.txt"), "w")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    writer = SummaryWriter(opt.save)
    
    device = f"cuda:{opt.local_rank}"
    init_seeds(seed=opt.seed)
    
    # load pre-trained VAE
    with open(os.path.join(opt.checkpoint, "opt.json"), "r") as f:
        opt_vae = json.load(f)
        opt_vae = Dict(opt_vae)
    ckpt = torch.load(os.path.join(opt.checkpoint, "best.ckpt"))
    netG = DVAE.DCGAN_G(opt_vae.imageSize, opt_vae.nz, opt_vae.nc, opt_vae.ngf, opt_vae.ngpu).to(device)
    netE = DVAE.Encoder(opt_vae.imageSize, opt_vae.nz, opt_vae.nc, opt_vae.ngf, opt_vae.ngpu).to(device)
    netG.load_state_dict(ckpt["netG"])
    netE.load_state_dict(ckpt["netE"])
    
    
    loader, _, num_classes = datasets.get_loaders(opt)
    
    if opt.dataset == 'cifar10':
        EBM_model = EBM_CIFAR32(3,opt.n_channel, data_init = opt.data_init).to(device)
    elif opt.dataset == 'mnist':
        EBM_model = EBM_CIFAR32(1,opt.n_channel, data_init = opt.data_init).to(device)
    elif opt.dataset == 'celeba_64':
        EBM_model = EBM_CelebA64(3,opt.n_channel, data_init = opt.data_init).to(device)
    elif opt.dataset == 'lsun_church':
        EBM_model = EBM_LSUN64(3,opt.n_channel, data_init = opt.data_init).to(device)
    elif opt.dataset == 'celeba_256':
        EBM_model = EBM_CelebA256(3,opt.n_channel, data_init = opt.data_init).to(device)
    else:
        raise Exception("choose dataset in ['cifar10', 'celeba_64', 'lsun_church', 'celeba_256']")
    
    # use 5 batches of training images to initialize the data dependent init for weight norm
    init_image = []
    for idx, (x, y) in enumerate(loader):
        init_image.append(x)
        B, C, W, H = x.size()
        if idx == 4:
            break
    init_image = torch.cat(init_image, axis=0).to(device)
    EBM_model(init_image)
    
    requires_grad(netE.parameters(), False)
    requires_grad(netG.parameters(), False)
    optimizer = optim.Adam(EBM_model.parameters(), lr=opt.lr, betas=(0.99, 0.999), weight_decay=opt.wd)
    
    noise_z = torch.empty((B, netG.nz), device=device)
    noise_x = torch.empty((B, C, W, H), device=device)
    
    for idx, (x, y) in tqdm(enumerate(sample_data(loader)), total=opt.total_iter):
        x = x.to(device)
        
        eps_z = torch.randn(opt.batch_size, netG.nz, device=device, requires_grad=True)
        eps_x = torch.randn_like(x, device=device, requires_grad=True)
        
        requires_grad(EBM_model.parameters(), False)
        EBM_model.eval()
        netE.eval()
        netG.eval()
        
        for k in range(opt.num_steps):
            # convert eps_z to B, nz, 1, 1
            dist_xz = netG(eps_z.reshape(opt.batch_size, netG.nz, 1, 1), in_vae=False)
            # B, C, W, H
            neg_x = dist_xz.sample(eps_x)
            
            # save_image(neg_x, os.path.join(opt.save, f"mcmc_{idx}_{k}.png"), nrow=16, normalize=True)
            
            # B
            log_pxgz = dist_xz.log_prob(neg_x).sum(dim=(1,2,3))
            
            # compute energy
            # 1/Z exp(-E(x)) p(x,h) = 1/Z exp(-F(x))
            # F(x) = E(x) - log p(x,h)
            # model(x) = E(x)
            # And the optimization objective is min E(neg_x) - E(pos_x)
            dvalue = EBM_model(neg_x) - log_pxgz
            dvalue = dvalue.mean()
            dvalue.backward()
            
            # update z
            noise_z.normal_(0, 1)
            eps_z.data.add_(eps_z.grad.data * opt.batch_size, alpha=-0.5*opt.step_size)
            eps_z.data.add_(noise_z.data, alpha=np.sqrt(opt.step_size))
            eps_z.grad.detach_()
            eps_z.grad.zero_()
            
            # update x
            noise_x.normal_(0, 1)
            eps_x.data.add_(eps_x.grad.data * opt.batch_size, alpha=-0.5*opt.step_size)
            eps_x.data.add_(noise_x.data, alpha=np.sqrt(opt.step_size))
            eps_x.grad.detach_()
            eps_x.grad.zero_()
        
        eps_z = eps_z.detach()
        eps_x = eps_x.detach()
        
        requires_grad(EBM_model.parameters(), True)
        EBM_model.train()
        EBM_model.zero_grad()
        
        dist_xz = netG(eps_z.reshape(opt.batch_size, netG.nz, 1, 1), in_vae=False)
        if opt.use_mu_cd:
            neg_x = 0.5*dist_xz.dist.mu + 0.5
        else: 
            neg_x = dist_xz.sample(eps_x)
        
        pos_out = EBM_model(x)
        neg_out = EBM_model(neg_x)
        
        norm_loss = EBM_model.spectral_norm_parallel()
        loss_reg_s = opt.alpha_s * norm_loss
        loss = pos_out.mean() - neg_out.mean()
        loss_total = loss + loss_reg_s
        
        loss_total.backward()
        
        if opt.grad_clip:
            clip_grad(EBM_model.parameters(), optimizer)

        optimizer.step()
        
        if (idx+1) % 100 == 0 or (idx+1) == opt.total_iter:
            logging.info(f"loss: {loss.item():.8f}")
            writer.add_scalar("loss", loss.item(), idx)
            
            neg_img = 0.5 * dist_xz.dist.mu + 0.5
            save_image(neg_img, os.path.join(opt.save, f"sample_iter_{idx}.png"), nrow=5, normalize=True)
            
            torch.save({
                "model": EBM_model.state_dict(),
                "optimizer": optimizer.state_dict()
                }, os.path.join(opt.save, f"current_{idx}.ckpt"))
        
        if loss < -10000 or idx >= opt.total_iter:
            break
        


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()
    
    
if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser('training of VAEBM')
    # # experimental results
    # parser.add_argument('--checkpoint', type=str, default='output/cifar10/20230421-214538',
    #                     help='location of the NVAE checkpoint')
    # parser.add_argument('--save', type=str, default='output/vaebm/',
    #                     help='location of the NVAE logging')

    # parser.add_argument('--dataset', type=str, default='cifar10',
    #                     help='which dataset to use')
    # parser.add_argument('--im_size', type=int, default=64, help='size of image')

    # parser.add_argument('--data', type=str, default='data/',
    #                     help='location of the data file')

    # parser.add_argument('--lr', type=float, default=5e-5,
    #                     help='learning rate for EBM')

    # # DDP.
    # parser.add_argument('--local_rank', type=int, default=1,
    #                     help='rank of process')
    # parser.add_argument('--world_size', type=int, default=1,
    #                     help='number of gpus')
    # parser.add_argument('--seed', type=int, default=1,
    #                     help='seed used for initialization')
    # parser.add_argument('--master_address', type=str, default='127.0.0.1',
    #                     help='address for master')
    
    # parser.add_argument('--batch_size', type=int, default = 32, help='batch size for training EBM')
    # parser.add_argument('--n_channel', type=int, default = 64, help='initial number of channels of EBM')

    # # traning parameters
    # parser.add_argument('--alpha_s', type=float, default=0.2, help='spectral reg coef')

    # parser.add_argument('--step_size', type=float, default=5e-6, help='step size for LD')
    # parser.add_argument('--num_steps', type=int, default=10, help='number of LD steps')
    # parser.add_argument('--total_iter', type=int, default=30000, help='number of training iteration')


    # parser.add_argument('--wd', type=float, default=3e-5, help='weight decay for adam')
    # parser.add_argument('--data_init', dest='data_init', action='store_false', help='data depedent init for weight norm')
    # parser.add_argument('--use_mu_cd', dest='use_mu_cd', action='store_true', help='use mean or sample from the decoder to compute CD loss')
    # parser.add_argument('--grad_clip', dest='grad_clip', action='store_false',help='clip the gradient as in Du et al.')    
    # parser.add_argument('--use_amp', dest='use_amp', action='store_true', help='use mix precision')
    
    # #buffer
    # parser.add_argument('--use_buffer', dest='use_buffer', action='store_true', help='use persistent training, default is false')
    # parser.add_argument('--buffer_size', type=int, default = 10000, help='size of buffer')
    # parser.add_argument('--max_p', type=float, default=0.6, help='maximum p of sampling from buffer')
    # parser.add_argument('--anneal_step', type=float, default=5000., help='p annealing step')
    
    # parser.add_argument('--comment', default='', type=str, help='some comments')
    
    # args = parser.parse_args()
    
    
    # args.distributed = False
    
    config = sys.argv[1] # tools/xx.json
    
    with open(config) as f:
        args = json.load(f)
    args = Dict(args)
    
    init_processes(0, 1, main, args)
    