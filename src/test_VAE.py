import argparse
import random
import os
import sys
import json
import time
import math

from addict import Dict
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.nn.functional import softmax

import utils
import DCGAN_VAE_pixel as DVAE

############# This module reconstruct and generate images from trained VAEs.


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,index)


def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) 
        return KL


def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        sigma = torch.exp(0.5*logvar)
        b = x.size(0)
        
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
        
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = (z - mu) / sigma
        z_eps = z_eps.view(b,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)
        
        weights = log_p_x_z+log_p_z-log_q_z_x
        
    return weights

def compute_NLL(weights):
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max())
        
    return NLL_loss


# python src/test_VAE.py --train_dir output/cifar10/20230421-214538 --dataset cifar10
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="directory of output from training")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument('--repeat', type=int, default=200, help='repeat for comute IWAE bounds')
    parser.add_argument('--num_iter', type=int, default=100, help='number of iters to optimize')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    
    opt = parser.parse_args()
    
    return opt


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


def main():
    if not torch.cuda.is_available():
        print("no gpu device available")
        sys.exit(1)
        
    test_opt = parse_args()
    with open(os.path.join(test_opt.train_dir, "opt.json")) as f:
        opt = Dict(json.load(f))
    # merge training and test params, overwrite training param with test param if overlapping
    for key, value in vars(test_opt).items():
        opt[key] = value
    
    device = "cuda:0"
    
    init_seeds(opt.seed, False)
    
    # opt.savedir = "{}/test-{}".format(opt.train_dir, time.strftime("%Y%m%d-%H%M%S"))
    opt.savedir = opt.train_dir
    utils.create_exp_dir(opt.savedir, scripts_to_save=None)
    with open(os.path.join(opt.savedir, "opt.json"), "w") as f:
        json.dump(opt, f, indent=4)
    print(opt)
    
    
    transform = transforms.Compose([transforms.Resize((opt.imageSize, opt.imageSize)), transforms.ToTensor()])
    # setup dataset
    if opt.dataset == "celeba_64":
        test_data = dset.CelebA(root=opt.datadir, split="valid", download=False, transform=transform)
    else:
        if opt.dataset == "fmnist":
            data_api = dset.FashionMNIST
        elif opt.dataset == "cifar10":
            data_api = dset.CIFAR10
        elif opt.dataset == "mnist":
            data_api = dset.MNIST
        else:
            raise NameError("Invalid dataset name")
    
        test_data = data_api(root=opt.datadir, train=True, download=True, transform=transform)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)
    
    
    # setup models
    netG = DVAE.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu).to(device)
    netE = DVAE.Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu).to(device)
    ckpt = torch.load(os.path.join(opt.train_dir, "best.ckpt"))
    netG.load_state_dict(ckpt["netG"])
    netE.load_state_dict(ckpt["netE"])
    netG.eval()
    netE.eval()
    
    # setup loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    path0 = os.path.join(opt.savedir, "original_batch")
    path1 = os.path.join(opt.savedir, "random")
    path2 = os.path.join(opt.savedir, "recon")
    path3 = os.path.join(opt.savedir, "random_batch")
    path4 = os.path.join(opt.savedir, "recon_batch")
    for p in [path0, path1, path2, path3, path4]:
        os.makedirs(p, exist_ok=True)
    
    for i, (xi, _) in enumerate(test_queue):
        if i >= 200:
            break
        # visualize original images
        grid = make_grid(xi, nrow=int(math.sqrt(opt.batchSize)), pad_value=1)
        save_image(grid, os.path.join(path0, f"{i:d}.jpg"))
        
        B = xi.size(0)
        # for j in range(B):
        #     save_image(xi[j], f"{opt.savedir}/{j+i*B}.jpg")
            
        
        xi = xi.to(device, non_blocking=True)
        b = xi.size(0)
        
        target = (xi.view(-1)*255).to(torch.int64)
        
        with torch.no_grad():
            ############# Reconstruct samples
            z, mu, logvar = netE(xi) # mu, logvar of q(z|x) ~ N(mu(x), logvar(x))
            # sample from q(z|x)
            # KL(q(z|x), Z) = f(mu, logvar)
            # B, 3, 32, 32
            distxz = netG(z, in_vae=True)
            # B, 3, 32, 32
            recon = distxz.sample(eps_x=0)
            grid = make_grid(recon, nrow=int(math.sqrt(opt.batchSize)), pad_value=1)
            save_image(grid, os.path.join(path4, f"{i:d}.jpg"))
            for j in range(B):
                save_image(recon[j], os.path.join(path2, f"{j+i*B}.jpg"))
            
            ############# Generated new samples
            z_rand = torch.randn_like(z)
            distxz = netG(z_rand, in_vae=True)
            # B, 3, 32, 32
            recon = distxz.sample(eps_x=0)
            grid = make_grid(recon, nrow=int(math.sqrt(opt.batchSize)), pad_value=1)
            save_image(grid, os.path.join(path3, f"{i:d}.jpg"))
            for j in range(B):
                save_image(recon[j], os.path.join(path1, f"{j+i*B}.jpg"))
            
            # ############## Perturb latent dimension for distanglement
            # limit = 4
            # interpolation = torch.arange(-limit, limit+0.1, 2*limit/10)
            
            # final_samples = []
            
            # splits = np.split(np.arange(netG.nz), netG.nz//5)
            
            # for val in interpolation:
            #     for split in splits:
            #         z_copy = z_rand.clone()
            #         z_copy.data[:,split] = val
            #         z_copy = z_copy.detach().clone().requires_grad_(True)
                    
            #         distxz = netG(z_copy, in_vae=True)
            #         # B, 3, 32, 32
            #         recon = distxz.sample(eps_x=0)
                    
            #         final_samples.append(recon)
                    
            # final_samples = torch.stack(final_samples, dim=1)
            # B, F1F2, C, W, H = final_samples.size()
            # print(final_samples.shape)
            # for j in range(B):
            #     grid = make_grid(final_samples[j], nrow=len(splits))
            #     save_image(grid, f"{opt.savedir}/{j+i*B}.jpg", normalize=True)


if __name__ == "__main__":
    main()