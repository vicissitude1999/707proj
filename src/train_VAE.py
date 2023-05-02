
import argparse
import random
import os
import sys
import logging
import json
import time
import glob
from tqdm import tqdm
from addict import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import utils
import DCGAN_VAE_pixel as DVAE
import distributions


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataroot", default="./data", help="path to dataset")
#     parser.add_argument("--seed", default=5, help="seed")
#     parser.add_argument("--workers", type=int, help="number of data loading workers", default=0)
#     parser.add_argument("--batchSize", type=int, default=128, help="input batch size")
#     parser.add_argument("--imageSize", type=int, default=32, help="the height / width of the input image to network")
#     parser.add_argument("--nc", type=int, default=1, help="input image channels")
#     parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
#     parser.add_argument("--ngf", type=int, default=32, help = "hidden channel sieze")
#     parser.add_argument("--niter", type=int, default=200, help="number of epochs to train for")
#     parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")

#     parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam. default=0.9")
#     parser.add_argument("--beta", type=float, default=1., help="beta for beta-vae")

#     parser.add_argument("--ngpu"  , type=int, default=1, help="number of GPUs to use")
#     parser.add_argument("--experiment", default=None, help="Where to store samples and models")
#     parser.add_argument("--perturbed", action="store_true", help="Whether to train on perturbed data, used for comparing with likelihood ratio by Ren et al.")
#     parser.add_argument("--ratio", type=float, default=0.2, help="ratio for perturbation of data, see Ren et al.")

#     opt = parser.parse_args()
    
#     return opt


def parse_args():
    config = sys.argv[1] # tools/xx.json
    
    with open(config) as f:
        opt = json.load(f)
    opt = Dict(opt)
    
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


def KL_div(mu,logvar,reduction = "avg"):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1) 
        return KL


def perturb(x, mu, device):
    b,c,h,w = x.size()
    mask = torch.rand(b,c,h,w, dtype=torch.float32, device=device) < mu
    noise = torch.FloatTensor(x.size()).random_(0, 256).to(device)
    x = x*255
    perturbed_x = ((1-mask)*x + mask*noise)/255.
    return perturbed_x


def main():
    if not torch.cuda.is_available():
        print("no gpu device available")
        sys.exit(1)
    opt = parse_args()
    print(opt)
    device = "cuda:0"
    
    # setup seed
    init_seeds(opt.seed, False)
    
    # setup output dir
    opt.savedir = os.path.join(opt.savedir, "vae", opt.dataset, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(opt.savedir, scripts_to_save=glob.glob("src/*.py"))
    with open(os.path.join(opt.savedir, "opt.json"), "w") as f:
        json.dump(opt, f, indent=4)
    
    # setup logging
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(opt.savedir, "log.txt"), "w")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    # setup tensorboard
    writer = SummaryWriter(opt.savedir)
    
    transform = transforms.Compose([transforms.Resize((opt.imageSize, opt.imageSize)), transforms.ToTensor()])
    
    # setup dataset
    if opt.dataset == "celeba64":
        train_data = dset.CelebA(root=opt.datadir, split="train", download=False, transform=transform)
    else:
        if opt.dataset == "fmnist":
            data_api = dset.FashionMNIST
        elif opt.dataset == "cifar10":
            data_api = dset.CIFAR10
        elif opt.dataset == "mnist":
            data_api = dset.MNIST
        else:
            raise NameError("Invalid dataset name")
    
        train_data = data_api(root=opt.datadir, train=True, download=True, transform=transform)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=opt.workers)
    
    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            
    # setup models
    netG = DVAE.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    netG.apply(weights_init)
    netE = DVAE.Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    netE.apply(weights_init)
    netE.to(device)
    netG.to(device)
    
    # setup optimizer
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay=3e-5)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=3e-5)
    
    # setup loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    best_obj = float("inf") # lowest training loss
    
    for epoch in tqdm(range(opt.niter)):
        logging.info(f"[epoch] {epoch:d}/{opt.niter}")
        # train
        objs = utils.AvgrageMeter()
        objs_rec = utils.AvgrageMeter()
        objs_kl = utils.AvgrageMeter()
        netE.train()
        netG.train()
        
        for step, (x, y) in enumerate(train_queue):
            x = x.to(device, non_blocking=True)
            
            if opt.perturbed:
                x = perturb(x, opt.ratio, device)
            
            b = x.size(0)
            
            [z, mu, logvar] = netE(x)
            # B, 3, 32, 32
            # recon = netG(z)
            # recl = torch.mean(torch.sum((recon - x)**2, axis=(1,2,3)))
            
            # B, 6, 32, 32
            dist_xz = netG(z, in_vae=True) # p(x|z)
            
            # B, C, W, H
            logp = dist_xz.log_prob(x)
            
            recl = torch.mean(- torch.sum(logp, dim=(1, 2, 3)))
            
            kld = KL_div(mu, logvar).mean()
            loss =  recl + opt.beta * kld
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            optimizer2.step()
            
            objs.update(loss.item(), b)
            objs_rec.update(recl.item(), b)
            objs_kl.update(kld.item(), b)
            
            if step % opt.report_freq == 0:
                writer.add_scalar("LossB/L", objs.avg, epoch * len(train_queue) + step)
                writer.add_scalar("LossB/rec", objs_rec.avg, epoch * len(train_queue) + step)
                writer.add_scalar("LossB/kl", objs_kl.avg, epoch * len(train_queue) + step)
                logging.info(f"train {step:03d}/{len(train_queue):03d} loss {objs.avg:.3f} rec {objs_rec.avg:.3f} kl {objs_kl.avg:.3f}")
        writer.add_scalar("Loss/L", objs.avg, epoch)
        writer.add_scalar("Loss/rec", objs_rec.avg, epoch)
        writer.add_scalar("Loss/kl", objs_kl.avg, epoch)
        writer.add_scalar("lr1", optimizer1.param_groups[0]["lr"], epoch)
        writer.add_scalar("lr2", optimizer2.param_groups[0]["lr"], epoch)
        
        train_obj = objs.avg
        logging.info(f"[train] loss {train_obj:.3f} rec {objs_rec.avg:.3f} kl {objs_kl.avg:.3f}")
        
        is_best = False
        if train_obj < best_obj:
            best_obj = train_obj
            is_best = True
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "best_obj": best_obj,
                "netE": netE.state_dict(),
                "netG": netG.state_dict(),
                "optimizer1": optimizer1.state_dict(),
                "optimizer2": optimizer2.state_dict()
            },
            is_best,
            opt.savedir,
        )


if __name__ == "__main__":
    main()
