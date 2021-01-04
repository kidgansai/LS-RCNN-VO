# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.utils.data import DataLoader
import re
import argparse
from time import time
from tensorboardX import SummaryWriter
from func import *
from net.cnn import Net
from dataset.KITTI_OF import KITTI_OF

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='Test', help='[Train / Test]')
parser.add_argument('--resume', default='Yes', help='[Yes / No] for cnn, [cnn / lstm / No] for cnn-lstm')
# 权重恢复的路径
parser.add_argument('--net_restore', default='cnn-vo', help='Restore net name')
parser.add_argument('--dir_restore', default='20200708', help='Restore time')
parser.add_argument('--model_restore', default='model-140', help='Restore model-id')
# 保存权重的路径
parser.add_argument('--net_name', default='cnn-vo', help='[cnn-vo /ae-vo/ rcnn-vo]')
parser.add_argument('--net_time', default='20201009', help='save time, such as 20200102')
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

def run_batch( datas,model,loss_func=None, optimizer=None, phase=None):
    if phase == 'Train':
        model.train()
    else:
        model.eval()  # 启用测试模式，关闭dropout

    of = to_var(datas['of'])
    label_pre = model(of)  # [32, 6]
    if phase == 'Train' or phase == 'Validate':
        label = to_var(datas['label'])  # [bs, 6]
        label = label.view(-1, 6)
        loss1 = loss_func(label_pre[:, :3], label[:, :3])
        loss2 = loss_func(label_pre[:, 3:], label[:, 3:])
        loss = loss1 + args.beta * loss2

        if phase == 'Train':
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # bp, compute gradients
            optimizer.step()  # apply gradients

        return loss.item(), loss1.item(), loss2.item(), label_pre.data
    else:
        return label_pre.data

def run_validate(model, loss_func, data_l):
    loss_ret = []
    loss1_ret = []
    loss2_ret = []
    for _, datas_v in enumerate(data_l):
        loss_v, loss1_v, loss2_v, _ = run_batch(datas=datas_v, model=model, loss_func=loss_func, phase='Validate')
        loss_ret.append(loss_v)
        loss1_ret.append(loss1_v)
        loss2_ret.append(loss2_v)
    loss_mean = np.mean(loss_ret)
    loss1_mean = np.mean(loss1_ret)
    loss2_mean = np.mean(loss2_ret)
    return loss_mean, loss1_mean, loss2_mean

def run_test(model, seq, model_nets_time=None, epoch=None, net_time_dir=None):
    print('\nTest sequence {:02d} >>>'.format(seq))
    data = KITTI_OF(data_dir=data_dir, label_dir=label_dir, phase='Test', seq=seq)
    data_l = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pose_rel = []
    for _, data_batch in enumerate(tqdm(data_l)):
        pose_pre = run_batch(datas=data_batch, model=model, phase='Test')
        pose_rel.extend(pose_pre.cpu().numpy())
    pose_abl = relative2absolute(pose_rel)

    if args.phase == 'Test':
        np.savetxt((net_time_dir + '/{:02d}.txt'.format(seq)), pose_abl)
        plot_pose(seq=seq, label_dir=label_dir, save_dir=net_time_dir, pose_abl=pose_abl, args=args)
        print('Save pose and trajectory in {:s}'.format(net_time_dir))
    else:
        plot_pose(seq=seq,label_dir=label_dir, save_dir=model_nets_time, pose_abl=pose_abl, epoch=epoch, args=args)
        print('Save trajectory in {:s}'.format(model_nets_time))

def main():
    torch.set_default_tensor_type('torch.FloatTensor')
    model = Net()
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu)

    # Set weights
    print('\n========================================')
    print('Phase: {:s}'.format(args.phase))
    if args.resume == 'Yes' or args.phase == 'Test':
        print('Restore from : {:s}'.format(restore_dir))
        model.load_state_dict(torch.load(restore_dir))
    else: print('Initialize from scratch')
    print('========================================')
    # Start training
    if args.phase == 'Train':
        model_nets_time, log_nets_time = folder_train(model_dir, log_dir, args)
        writer = SummaryWriter(log_nets_time)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr_base)
        data_t = KITTI_OF(data_dir=data_dir, label_dir=label_dir,  phase='Train')
        data_v = KITTI_OF(data_dir=data_dir, label_dir=label_dir,  phase='Validate')
        data_l_t = DataLoader(data_t, batch_size=args.batch_size, shuffle=True, num_workers=4)
        data_l_v = DataLoader(data_v, batch_size=args.batch_size, shuffle=False, num_workers=4)
        step_per_epoch = int(math.floor(len(data_t) / data_l_t.batch_size))
        step_validate = int(math.floor(step_per_epoch / 3))  # 每个epoch验证3次
        min_loss_v = 1e10
        for epoch in np.arange(args.epoch_max):
            adjust_learning_rate(optimizer, epoch, args.lr_base, args.lr_decay_rate, args.epoch_lr_decay)
            # 测试一个完整的序列
            if epoch != 0 and epoch % 5 == 0:
                run_test(model, seq=9, model_nets_time=model_nets_time, epoch=epoch)
                run_test(model, seq=5, model_nets_time=model_nets_time, epoch=epoch)

            loss_list = []  # 记录每个epoch的loss
            loss1_list = []
            loss2_list = []
            loss_v_mean_valid = 0
            for step, datas_t in enumerate(data_l_t):
                step_global = epoch * step_per_epoch + step
                tic = time()
                loss, loss1, loss2, _ = run_batch(datas=datas_t, model=model, loss_func=loss_func, optimizer=optimizer, phase='Train')
                hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)
                loss_list.append(loss)
                loss1_list.append(loss1)
                loss2_list.append(loss2)

                # display and add to tensor board
                if (step+1) % 10 == 0:
                    display_t(hour_per_epoch, epoch, args, step, step_per_epoch, optimizer, loss, loss1,
                                    loss2, loss_list, loss1_list, loss2_list, writer, step_global)
                    # run_test(model, seq=9, dir_model=dir_model, epoch=epoch)

                if (step+1) % step_validate == 0:
                    batch_v = int(math.ceil(len(data_v)/data_l_v.batch_size))
                    loss_v, loss1_v, loss2_v = run_validate(model, loss_func, data_l_v)
                    loss_v_mean_valid += float(loss_v)
                    display_v(batch_v, loss_v, loss1_v, loss2_v, args, writer, step_global)

            # save
            loss_v_mean_valid = loss_v_mean_valid/(len(loss_list))

            if loss_v_mean_valid < min_loss_v:
                min_loss_v = loss_v_mean_valid
                print('Save model at ep {}, mean of valid loss: {}'.format(epoch + 1, loss_v_mean_valid))
                print('\nSaving model_valid: {:s}/model-{:d}.pkl'.format(model_nets_time, epoch + 1))
                torch.save(model.state_dict(), (model_nets_time + '/model-validate.pkl'))

            if ((epoch+1) % 5 == 0) and (epoch < args.epoch_max):
                print('\nSaving model: {:s}/model-{:d}.pkl'.format(model_nets_time, epoch+1))
                torch.save(model.state_dict(), (model_nets_time + '/model-{:d}.pkl'.format(epoch+1)))
    else:
        net_time_dir = folder_test(test_dir,args)
        for seq in [3,6]:
            run_test(model, seq=seq, net_time_dir=net_time_dir)

if __name__ == '__main__':
    main()
