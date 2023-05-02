# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for ebm. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
'''Code for training ebm'''
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


def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        for i in range(iter):
            if i % 10 == 0:
                print('setting BN statistics iter %d out of %d' % (i+1, iter))
            model.train()
            model.sample(num_samples, t)
        model.eval()

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


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


def main(args):
    args.save = os.path.join(args.save)
    utils.create_exp_dir(args.save)
    with open(os.path.join(args.save, "args.json"), "w") as f:
        json.dump(args, f, indent=4)
    
    device = f"cuda:{args.local_rank}"
    # init_seeds(seed=args.seed)
    
    # load pretrained VAE
    with open(os.path.join(args.checkpoint_vae, "opt.json"), "r") as f:
        opt_vae = json.load(f)
        opt_vae = Dict(opt_vae)
    ckpt_vae = torch.load(os.path.join(args.checkpoint_vae, "best.ckpt"))
    netG = DVAE.DCGAN_G(opt_vae.imageSize, opt_vae.nz, opt_vae.nc, opt_vae.ngf, opt_vae.ngpu).to(device)
    netE = DVAE.Encoder(opt_vae.imageSize, opt_vae.nz, opt_vae.nc, opt_vae.ngf, opt_vae.ngpu).to(device)
    netG.load_state_dict(ckpt_vae["netG"])
    netE.load_state_dict(ckpt_vae["netE"])
    
    # load pretrained ebm
    with open(os.path.join(args.checkpoint_ebm, "args.json"), "r") as f:
        opt_ebm = json.load(f)
        opt_ebm = Dict(opt_ebm)
    ckpt_ebm = torch.load(os.path.join(args.checkpoint_ebm, f"current_{args.iter}.ckpt"))
    
    if args.dataset == 'cifar10':
        EBM_model = EBM_CIFAR32(3,opt_ebm.n_channel, data_init = opt_ebm.data_init).to(device)
    elif args.dataset == 'mnist':
        EBM_model = EBM_CIFAR32(1,opt_ebm.n_channel, data_init = opt_ebm.data_init).to(device)
    elif args.dataset == 'celeba_64':
        EBM_model = EBM_CelebA64(3,opt_ebm.n_channel, data_init = opt_ebm.data_init).to(device)
    elif args.dataset == 'lsun_church':
        EBM_model = EBM_LSUN64(3,opt_ebm.n_channel, data_init = opt_ebm.data_init).to(device)
    elif args.dataset == 'celeba_256':
        EBM_model = EBM_CelebA256(3,opt_ebm.n_channel, data_init = opt_ebm.data_init).to(device)
    else:
        raise Exception("choose dataset in ['cifar10', 'celeba_64', 'lsun_church', 'celeba_256']")
    
    # with torch.no_grad():
    #     EBM_model(torch.rand(10,3,args.im_size,args.im_size).cuda()) #for weight norm data dependent init
    
    EBM_model.load_state_dict(ckpt_ebm["model"])
    
    iter_needed = args.num_samples // args.batch_size 
    netE.eval()
    netG.eval()
    
    B = args.batch_size
    C = 3
    W = args.im_size
    H = args.im_size
    
    noise_z = torch.empty((B, netG.nz), device=device)
    noise_x = torch.empty((B, C, W, H), device=device)
    
    limit = 3
    interpolation = torch.arange(-limit, limit+0.1, 2/3)
    print(interpolation)
    
    # for i in range(iter_needed):
    #     x = torch.zeros(B, 3, W, H)
        
    #     eps_z = torch.randn(B, netG.nz, device=device, requires_grad=True)
        
    #     final_samples = []
        
    #     splits = np.split(np.arange(netG.nz), netG.nz//4)
    #     # splits = [np.array([0,1,2,3])]
        
    #     for split in splits:
    #         for val in interpolation:
    #             eps_z_cp = eps_z.clone()
    #             eps_z_cp.data[:,split] = val
    #             eps_z_cp = eps_z_cp.detach().clone().requires_grad_(True)
    #             eps_x = torch.randn_like(x, device=device, requires_grad=True)
                
    #             requires_grad(EBM_model.parameters(), False)
    #             EBM_model.eval()
    #             netE.eval()
    #             netG.eval()
                
    #             for k in range(args.num_steps):
    #                 dist_xz = netG(eps_z_cp.reshape(B, netG.nz, 1, 1))
    #                 neg_x = dist_xz.sample(eps_x)
                    
    #                 if args.renormalize:
    #                     neg_x_renorm = 2. * neg_x - 1.
    #                 else:
    #                     neg_x_renorm = neg_x

    #                 log_pxgz = dist_xz.log_prob(neg_x_renorm).sum(dim=(1,2,3))
                    
    #                 dvalue = EBM_model(neg_x) - log_pxgz
    #                 dvalue = dvalue.mean()
    #                 dvalue.backward()
                    
    #                 # update z
    #                 noise_z.normal_(0, 1)
    #                 eps_z_cp.data.add_(eps_z_cp.grad.data * B, alpha=-0.5*args.step_size)
    #                 eps_z_cp.data.add_(noise_z.data, alpha=np.sqrt(args.step_size))
    #                 eps_z_cp.grad.detach_()
    #                 eps_z_cp.grad.zero_()
                    
    #                 # update x
    #                 noise_x.normal_(0, 1)
    #                 eps_x.data.add_(eps_x.grad.data * B, alpha=-0.5*args.step_size)
    #                 eps_x.data.add_(noise_x.data, alpha=np.sqrt(args.step_size))
    #                 eps_x.grad.detach_()
    #                 eps_x.grad.zero_()
                
    #             eps_z_cp = eps_z_cp.detach()
    #             eps_x = eps_x.detach()
    #             dist_xz = netG(eps_z_cp.reshape(B, netG.nz, 1, 1))
    #             final_sample= dist_xz.dist.mu
                
    #             final_samples.append(final_sample)
        
    #     final_samples = torch.stack(final_samples, dim=1)
    #     B, F1F2, C, W, H = final_samples.size()
    #     # final_samples = final_samples.reshape(-1, len(splits), len(interpolation), C, W, H)
    #     print(final_samples.shape)
    #     for j in range(B):
    #         grid = make_grid(final_samples[j], nrow=len(splits))
    #         save_image(grid, f"{args.save}/{j+i*B}.png", normalize=True)
    
    
    transform = transforms.Compose([transforms.Resize((W, H)), transforms.ToTensor()])
    test_data = dset.CelebA(root=opt_vae.datadir, split="valid", download=False, transform=transform)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
    
    path1 = f"{args.save}/mulog_added"
    path4 = f"{args.save}/mulog_added_batch"
    
    final_random_zs = []
    
    
    def mcmc(eps_z, eps_x):
        requires_grad(EBM_model.parameters(), False)
        EBM_model.eval()
        netE.eval()
        netG.eval()
        
        for k in range(args.num_steps):
            dist_xz = netG(eps_z.reshape(B, netG.nz, 1, 1))
            neg_x = dist_xz.sample(eps_x)
            
            if args.renormalize:
                neg_x_renorm = 2. * neg_x - 1.
            else:
                neg_x_renorm = neg_x

            log_pxgz = dist_xz.log_prob(neg_x_renorm).sum(dim=(1,2,3))
            
            dvalue = EBM_model(neg_x) - log_pxgz
            dvalue = dvalue.mean()
            dvalue.backward()
            
            # update z
            noise_z.normal_(0, 1)
            eps_z.data.add_(eps_z.grad.data * B, alpha=-0.5*args.step_size)
            eps_z.data.add_(noise_z.data, alpha=np.sqrt(args.step_size))
            eps_z.grad.detach_()
            eps_z.grad.zero_()
            
            # update x
            noise_x.normal_(0, 1)
            eps_x.data.add_(eps_x.grad.data * B, alpha=-0.5*args.step_size)
            eps_x.data.add_(noise_x.data, alpha=np.sqrt(args.step_size))
            eps_x.grad.detach_()
            eps_x.grad.zero_()
        
        eps_z = eps_z.detach()
        eps_x = eps_x.detach()
        
        return eps_z, eps_x
    
    
    method = "random"
    
    for i, (x, _) in enumerate(tqdm(test_queue, total=200)):
        if i >= 200:
            break
        x = x.to(device)
        # print(x.shape)
        z, mu, log_var = netE(x)
        # x = torch.zeros(B, 3, W, H)
        if method == "random":
            eps_z = torch.randn(B, netG.nz, device=device, requires_grad=True)
            eps_x = torch.randn_like(x, device=device, requires_grad=True)
        elif method == "mulog_mcmc": # z-only MCMC
            eps_z = z.squeeze().detach().clone().requires_grad_()
            eps_x = torch.randn_like(x, device=device, requires_grad=True)
        elif method == "mulog_both_mcmc": # z-and-x MCMC
            eps_z = z.squeeze().detach().clone().requires_grad_()
            eps_x = netG(eps_z.reshape(B, netG.nz, 1, 1)).dist.mu
            eps_x = eps_x.detach().clone().requires_grad_()
        else:
            raise NameError("Incorrect method")
        
        eps_z, eps_x = mcmc(eps_z, eps_x)
        
        final_random_zs.append(eps_z.cpu().numpy())
        dist_xz = netG(eps_z.reshape(B, netG.nz, 1, 1))
        final_sample = dist_xz.dist.mu
        
        path_1 = f"{args.save}/{method}"
        path_2 = f"{args.save}/{method}_batch"
        os.makedirs(path_1, exist_ok=True)
        os.makedirs(path_2, exist_ok=True)
        grid = make_grid(final_sample, nrow=4, pad_value=1)
        save_image(grid, os.path.join(path_2, f"{i}.jpg"), normalize=True)
        for j in range(B):
            save_image(final_sample[j], os.path.join(path_1, f"{i*B+j}.jpg"), normalize=True)
            
    final_random_zs = np.concatenate(final_random_zs, axis=0)
    np.save(f"{args.save}/final_random_zs.npy", final_random_zs)
    def multivar_kl_from_standard(mu, var):
        return 0.5*(np.trace(var)+np.linalg.norm(mu)-len(mu)-np.log(np.linalg.det(var)))
    kl = multivar_kl_from_standard(np.mean(final_random_zs, axis=0), np.cov(final_random_zs, rowvar=False))
    print(f'EBM sampled zs KL = {kl}')



if __name__ == '__main__':
    config = sys.argv[1] # tools/xx.json
    
    with open(config) as f:
        args = json.load(f)
    args = Dict(args)
    
    main(args)
    