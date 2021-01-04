# -*- coding: utf-8 -*-

import torch
import shutil
import os
import math
import numpy as np
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
from numpy import mat
from tqdm import tqdm
import torch.nn.functional as F

def folder_train(model_dir, log_dir, args):
    model_nets = model_dir + '/' + args.net_name
    log_nets = log_dir + '/' + args.net_name
    model_nets_time = model_nets + '/' + args.net_time
    log_nets_time = log_nets + '/' + args.net_time
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(model_nets):
        os.mkdir(model_nets)
    if not os.path.exists(log_nets):
        os.mkdir(log_nets)
    if not os.path.exists(model_nets_time):
        os.mkdir(model_nets_time)
    if os.path.exists(log_nets_time):
        shutil.rmtree(log_nets_time)
    os.mkdir(log_nets_time)
    return model_nets_time, log_nets_time

def folder_test(test_dir,args):
    net_dir = test_dir + '/' + args.net_restore
    net_time_dir = net_dir + '/' + args.dir_restore + '-' + args.model_restore
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(net_dir):
        os.mkdir(net_dir)
    if not os.path.exists(net_time_dir):
        os.mkdir(net_time_dir)
    return net_time_dir

def to_var(x):
    if torch.cuda.is_available():
        return Variable(x).cuda()
    else:
        return Variable(x)

def adjust_learning_rate(optimizer, epoch, lr_base, gamma=0.316, epoch_lr_decay=25):
    exp = int(math.floor(epoch / epoch_lr_decay))
    lr_decay = gamma ** exp
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_decay * lr_base

def display_t(hour_per_epoch, epoch, args, step, step_per_epoch, optimizer, loss, loss1, loss2,
                    loss_list, loss1_list, loss2_list, writer, step_global):
    print('time:{:.3f} epoch:[{:03d}/{:03d}] step:[{:03d}/{:03d}] lr:{:.7f} loss:{:.4f}({:.4f})={:.4f}({:.4f})+{:d}*{:.4f}({:.4f})'.format(hour_per_epoch, epoch + 1, args.epoch_max, step + 1,
    step_per_epoch,optimizer.param_groups[0]['lr'],loss,np.mean(loss_list), loss1,np.mean(loss1_list), args.beta, loss2, np.mean(loss2_list)))
    writer.add_scalars('./train-val', {'loss_t': loss, 'loss1_t': loss1, 'loss2_t': loss2}, step_global)

def display_v(batch_v, loss_v, loss1_v, loss2_v, args, writer, step_global):
    print('{:d}batches loss:{:.4f}={:.4f}+{:d}*{:.4f}'.format(batch_v, loss_v, loss1_v, args.beta, loss2_v))
    writer.add_scalars('./train-val', {'loss_v': loss_v, 'loss1_v': loss1_v, 'loss2_v': loss2_v}, step_global)

def eulerAnglesToRotationMatrix(theta):
    # 欧拉角转换为旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def relative2absolute(rel):
    rel = np.array(rel) #6-d
    abl = []  # 12-d
    t1 = mat(np.eye(4))
    abl.extend([np.array(t1[0: 3, :]).reshape([-1])])
    for i in tqdm(range(len(rel))):
        x12 = rel[i, 0]
        y12 = rel[i, 1]
        z12 = rel[i, 2]
        theta = np.array([rel[i, 3] / 180 * math.pi,rel[i, 4] / 180 * math.pi,rel[i, 5] / 180 * math.pi])
        Rot = eulerAnglesToRotationMatrix(theta)
        t12 = np.row_stack((np.column_stack((Rot, [[x12], [y12], [z12]])), [0, 0, 0, 1]))
        t2 = t1 * t12
        abl.extend([np.array(t2[0: 3, :]).reshape([-1])])
        t1 = t2
    return np.array(abl)

def plot_pose(seq,label_dir, save_dir, pose_abl, epoch=None, args=None):
    plt.close('all')
    style.use("ggplot")
    pose_gt = np.loadtxt(label_dir+'-12d/{:02d}.txt'.format(seq))
    pose_pre = np.array(pose_abl)
    plt.plot(pose_gt[:, 3], pose_gt[:, 11], '--', c='k',  label='Ground Truth')
    if args.phase == 'Train':
        plt.plot(pose_pre[:, 3], pose_pre[:, 11], '-', c='b',  label='model-{:d}'.format(epoch))
    else:
        plt.plot(pose_pre[:, 3], pose_pre[:, 11], '-', c='r',  label=args.model_restore)
    plt.title('Sequence {:02d}'.format(seq))
    plt.xlabel('x[m]',fontsize=12)
    plt.ylabel('z[m]',fontsize=12)
    plt.axis('equal')
    plt.legend(fontsize=12)
    if args.phase == 'Train':
        plt.savefig(save_dir+'/{:02d}-epoch-{:d}.png'.format(seq, epoch))
    else:
        plt.savefig(save_dir + '/{:02d}.png'.format(seq))

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def sparse_max_pool(input, size):
    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output

def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):
        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(output, target_scaled, sparse, mean=False)
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))
    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss

def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def read_flo(of_path):
    f = open(of_path, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    if 202021.25 != magic:
        print ('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def read_of(of_path):
    f = open(of_path, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        flow = np.resize(data2d, (h, w, 2))
    f.close()
    return flow