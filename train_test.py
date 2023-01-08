import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.SeaNet_models import SeaNet
from data import get_loader
from data import test_dataset
from utils import clip_gradient, adjust_lr
from scipy import misc
import time

import pytorch_iou

CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
IOU = pytorch_iou.IOU(size_average = True)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)

def _get_adaptive_threshold(matrix, max_value = 1):
    """
    Return an adaptive threshold, which is equal to twice the mean of ``matrix``.
    :param matrix: a data array
    :param max_value: the upper limit of the threshold
    :return: min(2 * matrix.mean(), max_value)
    """
    return min(2 * matrix.mean(), max_value)

def cal_adaptive_fm(pred, gt):
    """
    Calculate the adaptive F-measure.
    :return: adaptive_fm
    """
    # ``np.count_nonzero`` is faster and better
    beta = 0.3
    adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
    binary_predcition = (pred >= adaptive_threshold).astype(np.float32)
    area_intersection = np.count_nonzero(binary_predcition * gt)
    # area_intersection = torch.sum(torch.sum(binary_predcition & gt, dim=1), dim=1).float()
    if area_intersection == 0:
        adaptive_fm = 0
    else:
        pre = area_intersection * 1.0 / np.count_nonzero(binary_predcition)
        rec = area_intersection * 1.0 / np.count_nonzero(gt)
        adaptive_fm = (1 + beta) * pre * rec / (beta * pre + rec)
    return adaptive_fm

def run(train_i):
    best_adp_fm = 0
    best_mae = 1
    best_epoch = 0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
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
            loss4 = MSE(edge1, edge2) / (opt.trainsize * opt.trainsize)
            # torch 1.9.0
            # loss4 = MSE(edge1, edge2)
            loss = loss1 + loss2 + loss3 + 0.5 * loss4

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if i % 20 == 0 or i == total_step:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step,
                               opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                               loss2.data, loss3.data, 0.5 * loss4.data))
        save_path = 'models/' + str(train_i) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res_save_path = save_path + 'salmap/'
        if not os.path.exists(res_save_path):
            os.makedirs(res_save_path)

        # test
        with torch.no_grad():
            model.eval()
            time_sum = 0
            adaptive_fms = 0.0
            mae = 0.0
            for i in range(test_loader.size):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                time_start = time.time()
                res, s34, s5, s12_sig, s34_sig, s5_sig, edge1, edge2 = model(image)
                time_end = time.time()
                time_sum = time_sum + (time_end - time_start)
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # adaptive_fm = cal_adaptive_fm(pred=res, gt=gt)
                adaptive_fm = 0
                adaptive_fms += adaptive_fm
                mae += np.sum(np.abs(gt - res)) / (gt.shape[0] * gt.shape[1])
                misc.imsave(res_save_path + name, res)

            print('FPS {:.5f}'.format(test_loader.size / time_sum))
            adp_fm = adaptive_fms / test_loader.size
            mae_mean = mae / test_loader.size
            if mae_mean < best_mae:
                best_adp_fm = adp_fm
                best_mae = mae_mean
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'RGBT_SOD.pth', _use_new_zipfile_serialization=False)
                print('Epoch [{:03d}], best_adp_fm {:.4f}, best_mae {:.4f}'.format(epoch, best_adp_fm, best_mae))
            print('Current_epoch [{:03d}], adp_fm {:.4f}, mae {:.4f}'.format(epoch, adp_fm, mae_mean))
            print('Best_epoch [{:03d}], best_adp_fm {:.4f}, best_mae {:.4f}'.format(best_epoch, best_adp_fm, best_mae))

print("Let's go!")
for train_i in range(1, 20):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=288, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    opt = parser.parse_args()
    # build train_models
    model = SeaNet()
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    image_root = './dataset/train_dataset/EORSSD/train/image/'
    gt_root = './dataset/train_dataset/EORSSD/train/GT/'
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    # build test_models
    test_dataset_path = './dataset/test_dataset/'
    test_datasets = 'EORSSD'
    test_image_root = test_dataset_path + test_datasets + '/image/'
    test_gt_root = test_dataset_path + test_datasets + '/GT/'
    test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)

    print('Statr {}-th training!!!'.format(train_i))
    print('Learning Rate: {}'.format(opt.lr))

    run(train_i)

