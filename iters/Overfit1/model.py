import os
import sys
import csv
import time
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from random import shuffle
from torch.cuda import amp
from tqdm import tqdm
from PIL import Image

user = '/'.join(os.getcwd().split('/')[:3])
sys.path.insert(0, user+'/code/pycode/ML/VAE')
sys.path.insert(0, user+'/code/pycode/ML/VAE/models/networks')
from disc import *
import ResUnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

cwd = os.getcwd()
args = sys.argv
random.seed(3)
torch.manual_seed(3)

name = args[1].split('/')[-1]
training_mode = (args[2]=='train')
writer = SummaryWriter(log_dir='../../runs/'+name)

# Hyperparameters
batch_size = 120
learning_rate = 0.01
iterations = 10000
latent_dim_size = 4
imgs_shown = 3
beta = .9
prints_per_epoch = 10

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('~/data/mnist', train=True, download=True, transform=torchvision.transforms.Compose([T.ToTensor()])), batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('~/data/mnist', train=False, download=True, transform=torchvision.transforms.Compose([T.ToTensor()])), batch_size=batch_size, shuffle=True)

model = ResUnet.VAE(latent_dim_size).cuda()

if ('params.pth' in os.listdir()) and (not training_mode):
    model.load_state_dict(torch.load('params.pth'))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
crit = nn.MSELoss(reduction='sum')

total_params = sum(param.numel() for param in model.parameters())
print('Total params :', total_params)
disc_text("```"+name+' : '+str(total_params)+' parameters```', 'train1')
disc_text("```"+name+' : '+str(total_params)+' parameters```', 'test1')

def restore(img):
    invTrans = T.Compose([T.Normalize((0,),(1/0.3081,)), T.Normalize((-.1307,),(1.,))])
    return invTrans(img).numpy()

def save_and_test(epoch, data_loader):
    test_imgs = next(iter(data_loader))[0].cuda()
    total_imgs = test_imgs.shape[0]

    out = model(test_imgs).cpu().detach()
    test_imgs = test_imgs.cpu().detach()

    loss = (beta * crit(out, test_imgs) + (1-beta) * model.encoder.kl)/total_imgs
    
    #out = restore(out)
    #test_imgs = restore(test_imgs)
    test_imgs = test_imgs[:imgs_shown]

    imgs = np.array([np.hstack((test_imgs[i], out[i])) for i in range(imgs_shown)])

    #writer.add_image('Test Reconstructions', torchvision.utils.make_grid(torch.tensor(imgs)), epoch)
    imgs = torchvision.utils.make_grid(torch.tensor(imgs))
    imgs = T.ToPILImage()(imgs)
    imgs.save('outputs/epoch'+str(epoch)+'.png')
    imgs.save('tmp.png')
    disc_image('tmp.png', 'test1')
    writer.add_scalar('Test Loss', loss, epoch)
    writer.close()

    output = f'Testing loss on Epoch {epoch:3d} : {loss:.3f}'
    print(output)
    disc_text(output, 'test1')


def train(start_epoch):
    model.train()

    for epoch in range(start_epoch+1, iterations):
        total_num = 0
        current_loss = 0.0

        num_iters = len(train_loader)
        screen = num_iters//prints_per_epoch

        for i, (imgs, label) in tqdm(enumerate(train_loader), total=num_iters, leave=False):
            total_num += imgs.shape[0]
            optimizer.zero_grad()

            imgs = imgs.cuda()
            out = model(imgs)

            loss = beta * crit(out, imgs) + (1-beta) * model.encoder.kl
            current_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i+1)%screen == 0:
              output = f'[{epoch:3d}, {i//screen}] : {current_loss/total_num:.3f}'
              print(output)
              disc_text(output, 'train1')
              if training_mode:
                  with open('results.txt', 'a') as f:
                      f.write(output+'\n')
                  with open('.last-epoch', 'w') as f:
                      f.write(str(epoch+1))
                  torch.save(model.state_dict(), 'params.pth')
              writer.add_scalar('Train Loss', current_loss/total_num, epoch*prints_per_epoch+i//screen)
              writer.close()


        save_and_test(epoch, test_loader)


train(int(args[3]) if len(args)>=4 else -1)
