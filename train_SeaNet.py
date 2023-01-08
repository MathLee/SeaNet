import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.SeaNet_models import SeaNet
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=288, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()

# build models
model = SeaNet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
#
# image_root = './dataset/train_dataset/ORSSD/train/image/'
# gt_root = './dataset/train_dataset/ORSSD/train/GT/'
image_root = './dataset/train_dataset/EORSSD/train/image/'
gt_root = './dataset/train_dataset/EORSSD/train/GT/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)


CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
IOU = pytorch_iou.IOU(size_average = True)


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        s12, s34, s5, s12_sig, s34_sig, s5_sig, edge1, edge2 = model(images)

        loss1 = CE(s12, gts) + IOU(s12_sig, gts)
        loss2 = CE(s34, gts) + IOU(s34_sig, gts)
        loss3 = CE(s5, gts) + IOU(s5_sig, gts)
        # torch 0.4.0
        loss4 = MSE(edge1, edge2)/(opt.trainsize*opt.trainsize)
        # torch 1.9.0
        # loss4 = MSE(edge1, edge2)

        loss = loss1 + loss2 + loss3 + 0.5*loss4

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                           loss2.data, loss3.data, 0.5*loss4.data))


    save_path = 'models/SeaNet/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'SeaNet.pth' + '.%d' % epoch)

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
