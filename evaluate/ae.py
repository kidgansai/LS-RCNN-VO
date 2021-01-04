# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.utils.data import DataLoader
import re
import cv2
import argparse
from time import time
from tensorboardX import SummaryWriter
from func import *
from net.ae import Net
from dataset.KITTI_OF import KITTI_OF

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='Test', help='[Train / Test]')
parser.add_argument('--resume', default='Yes', help='[Yes / No] for cnn, [cnn / lstm / No] for cnn-lstm')
# 权重恢复的路径
parser.add_argument('--net_restore', default='ae', help='Restore net name')
parser.add_argument('--dir_restore', default='20201010', help='Restore time')
parser.add_argument('--model_restore', default='model-15_train', help='Restore model-id')
# 保存权重的路径
parser.add_argument('--net_name', default='ae', help='[cnn-vo /ae/ ae-vo/ rcnn-vo]')
parser.add_argument('--net_time', default='20201010', help='save time, such as 20200102')
# 训练参数
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--epoch_max', default=250, type=int, help='Max epoch')
parser.add_argument('--lr_base', default=1e-4, type=float, help='Base learning rate')
parser.add_argument('--lr_decay_rate', default=0.318, type=float, help='Decay rate of lr')
parser.add_argument('--epoch_lr_decay', default=30, type=int, help='Every # epoch, lr decay lr_decay_rate')
parser.add_argument('--beta', default=10, type=int, help='loss = loss_t + beta * loss_r')
# 多GPU训练的参数
parser.add_argument("--gpu", default='0', help='GPU id list')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu           # 设置可见的gpu的列表，例如：'2,3,4'
gpu_list = re.split('[, ]', args.gpu)                   # 提取出列表中gpu的id
args.gpu = range(len(list(filter(None, gpu_list))))     # 传给PyTorch中多gpu并行的列表

# 数据集、标签等所有路径
data_dir = 'e:/sai/kitti_flow_pwc/'
label_dir = 'D:/sai/LS-RCNN-VO/dataset/label/kitti-gt-6d'
model_dir = 'D:/sai/LS-RCNN-VO/model'
test_dir = 'D:/sai/LS-RCNN-VO/test'
log_dir = 'D:/sai/LS-RCNN-VO/log'
restore_dir = model_dir + '/' + args.net_restore + '/' + args.dir_restore + '/' + args.model_restore + '.pkl'

def run_batch(datas, model, optimizer=None, phase=None):
    if phase == 'Train':
        model.train()
    else:
        model.eval()  # 启用测试模式，关闭dropout
    of = to_var(datas['of'])
    output = model(of)
    loss = multiscaleEPE(output, of, weights=[0.005,0.01,0.02,0.08,0.32], sparse=False)
    flow2_EPE = realEPE(output[0], of, sparse=False)

    if phase == 'Train':
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # bp, compute gradients
        optimizer.step()  # apply gradients
    return loss.item(),flow2_EPE.item(),output[0].data

def run_test(model, seq,  net_time_dir=None):
    print('\nTest sequence {:02d} >>>'.format(seq))
    data = KITTI_OF(data_dir=data_dir, label_dir=label_dir, phase='Test', seq=seq)
    data_l = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
    for _, data_batch in enumerate(tqdm(data_l)):
        loss, flow_EPE, flow_pre = run_batch(datas=data_batch, model=model, phase='Test')
        flow_pre = flow_pre[0].cpu().numpy()
        flow_pre = np.transpose(flow_pre, [1, 2, 0])  # 6xHxW
        # 可视化需要将光流转换为(h,w,2)
        flow_img = flow_to_image(flow_pre)
        cv2.imwrite(net_time_dir + '/{:02d}/{:06d}.png'.format(seq,_), flow_img)
    print('Save predicted optical flow in {:s}'.format(net_time_dir))

def main():
    torch.set_default_tensor_type('torch.FloatTensor')
    model = Net()
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu)

    # Set weights
    print('\n========================================')
    print('Phase: {:s}\nNet architecture: {:s}'.format(args.phase, args.net_name))
    if args.resume == 'Yes' or args.phase == 'Test':
        print('Restore from CNN: {:s}'.format(restore_dir))
        model.load_state_dict(torch.load(restore_dir))
    else:
        print('Initialize from scratch')
    print('========================================')

    # Start training
    if args.phase == 'Train':
        model_nets_time, log_nets_time = folder_train(model_dir, log_dir, args)
        writer = SummaryWriter(log_nets_time)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)
        data_t = KITTI_OF(data_dir=data_dir, label_dir=label_dir, phase='Train')
        data_l_t = DataLoader(data_t, batch_size=args.batch_size, shuffle=True, num_workers=4)
        step_per_epoch = int(math.floor(len(data_t) / data_l_t.batch_size))
        min_loss_t = 1e10
        for epoch in np.arange(args.epoch_max):
            adjust_learning_rate(optimizer, epoch, args.lr_base, args.lr_decay_rate, args.epoch_lr_decay)
            flow_EPE_list = []  # 记录每个epoch的loss
            loss_list = []
            loss1_list = []
            loss2_list = []
            for step, datas_t in enumerate(data_l_t):
                step_global = epoch * step_per_epoch + step
                tic = time()
                loss,flow_EPE,_ = run_batch(datas=datas_t, model=model,  optimizer=optimizer, phase='Train')
                # 这里的loss1和loss2无意义，取空
                loss1 = 0
                loss2 = 0
                loss1_list.append(loss1)
                loss2_list.append(loss2)
                hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)
                flow_EPE_list.append(flow_EPE)
                loss_list.append(loss)
                # display and add to tensor board
                if (step + 1) % 10 == 0:
                    print(flow_EPE)
                    display_t(hour_per_epoch, epoch, args, step, step_per_epoch, optimizer, loss, loss1,
                                    loss2, loss_list, loss1_list, loss2_list, writer, step_global)
            # save if the training loss decrease
            loss_mean_train = np.mean(flow_EPE_list)
            if loss_mean_train < min_loss_t:
                min_loss_t = loss_mean_train
                print('Save model at ep {}, mean of train loss: {}'.format(epoch + 1, loss_mean_train))
                print('\nSaving model_train: {:s}/model-{:d}.pkl'.format(model_nets_time, epoch + 1))
                torch.save(model.state_dict(), (model_nets_time + '/model-{:d}_train.pkl'.format(epoch + 1)))

                if ((epoch + 1) % 5 == 0) and (epoch < args.epoch_max):
                    print('\nSaving model: {:s}/model-{:d}.pkl'.format(model_nets_time, epoch + 1))
                    torch.save(model.state_dict(), (model_nets_time + '/model-{:d}.pkl'.format(epoch + 1)))
    else:
        net_time_dir = folder_test(test_dir, args)
        for Seq in range(11):
            if not os.path.exists(net_time_dir+'/{:02d}'.format(Seq)):
                os.mkdir(net_time_dir+'/{:02d}'.format(Seq))
            run_test(model, seq=Seq, net_time_dir=net_time_dir)

if __name__ == '__main__':
    main()
