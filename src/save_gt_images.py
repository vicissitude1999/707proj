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


def main():
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    # setup dataset
    test_data = dset.CelebA(root="/data/10707project/", split="valid", download=False, transform=transform)
    
    
    os.makedirs("/data/10707project/output/groundtruth", exist_ok=True)
    for i in range(200*16):
        save_image(test_data[i][0], os.path.join(f"/data/10707project/output/groundtruth/{i:d}.jpg"))
        
if __name__ == "__main__":
    main()