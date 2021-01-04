# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.utils.data import DataLoader
import re
import argparse
from time import time
from tensorboardX import SummaryWriter
from func import *
from glob import glob
from net.cnn_lstm import Net
from dataset.KITTI_OF_LSTM import KITTI_OF_LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='Test', help='[Train / Test]')
parser.add_argument('--resume', default='cnn', help=' [cnn / Yes / No] for cnn-lstm')
# 权重恢复的路径
parser.add_argument('--net_restore', default='rcnn-vo', help='Restore net name')
parser.add_argument('--dir_restore', default='20201010', help='Restore time')
parser.add_argument('--model_restore', default='model-5', help='Restore model-id')
# 保存权重的路径
parser.add_argument('--net_name', default='rcnn-vo', help='[cnn-vo /ae-vo/ rcnn-vo]')
parser.add_argument('--net_time', default='20201011', help='save time, such as 20200102')
# 训练参数
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--time_step', default=4, type=int, help='time-step for lstm')
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

def run_batch(datas, model, loss_func=None, optimizer=None , phase=None):

    if phase == 'Train':
        model.train()
    else:
        model.eval()  # 启用测试模式，关闭dropout
    loss_mean = []
    loss1_mean = []
    loss2_mean = []
    of = to_var(datas['of'])
    label_pre = model(of)  #
    label = to_var(datas['label'])  # [bs,ip, 6]
    label = label.view(-1, 6)                #[4,6]
    loss1 = loss_func(label_pre[:, :3], label[:, :3])
    loss2 = loss_func(label_pre[:, 3:], label[:, 3:])
    loss = loss1 + args.beta * loss2

    loss_mean.append(loss)
    loss1_mean.append(loss1)
    loss2_mean.append(loss2)
    if phase == 'Train':
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # bp, compute gradients
        optimizer.step()  # apply gradients
    loss_mean = torch.mean(torch.stack(loss_mean))
    loss1_mean = torch.mean(torch.stack(loss1_mean))
    loss2_mean = torch.mean(torch.stack(loss2_mean))
    # print(loss_mean)
    return loss_mean.item(), loss1_mean.item(), loss2_mean.item()

def run_val(model, loss_func,loader):
    loss = []
    loss1 = []
    loss2 = []
    for _, datas_v in enumerate(loader):
        loss_v, loss1_v, loss2_v, = run_batch(datas=datas_v, model=model, loss_func=loss_func, phase='Validate')
        loss.append(loss_v)
        loss1.append(loss1_v)
        loss2.append(loss2_v)
    loss_mean = np.mean(loss)
    loss1_mean = np.mean(loss1)
    loss2_mean = np.mean(loss2)
    return loss_mean, loss1_mean, loss2_mean

def run_test(model, seq, model_nets_time= None, epoch=None, net_time_dir= None):
    print('\nTest sequence {:02d} >>>'.format(seq))
    model.eval()
    # 法一 速度更快一点
    of_list = glob(data_dir + '/{:02d}/*.flo'.format(seq))
    of_list.sort()
    ts = args.time_step
    temp_1 = int(math.floor(len(of_list) / ts))
    temp_2 = int(math.ceil(len(of_list) / ts))
    pose_rel = []
    for i in tqdm(np.arange(temp_1)):
        input_of = []
        # path = []
        for of_path in of_list[i * ts: (i + 1) * ts]:
            of = read_of(of_path)
            input_of.append(of)
        x = np.stack(input_of, 0)
        x = np.transpose(x, [0, 3, 1, 2])  # [4, C, H, W]
        x = x[np.newaxis, :, :, :, :]  # [1, 4, C, H, W]
        x = to_var(torch.from_numpy(x))
        predict_pose = model(x)
        pose_rel.extend(predict_pose.data.cpu().numpy())

    ss = temp_1 * ts
    if temp_1 != temp_2:
        # 对剩下的序列做一次性处理
        print('Process for the last {:d} images...'.format(len(of_list) - ss))
        input_of = []
        for of_path in of_list[ss:]:
            of = read_of(of_path)
            input_of.append(of)
        x = np.stack(input_of, 0)
        x = np.transpose(x, [0, 3, 1, 2])  # [4, C, H, W]
        x = x[np.newaxis, :, :, :, :]  # [1, 4, C, H, W]
        x = to_var(torch.from_numpy(x))
        predict_pose = model(x)
        pose_rel.extend(predict_pose.data.cpu().numpy())

#  法二：滚动式输出结果
#     of_list = glob(data_dir + '/{:02d}/*.flo'.format(seq))
#     of_list.sort()
#     ts = args.time_step
#     pose_rel = []
#     for i in tqdm(np.arange(len(of_list)-ts+1)):
#         input_of = []
#         for of_path in of_list[i: i + ts]:
#             of = read_of(of_path)
#             input_of.append(of)
#         # print(np.array(input_of).shape)
#         x = np.stack(input_of, 0)
#         x = np.transpose(x, [0, 3, 1, 2])  # [4, C, H, W]
#         x = x[np.newaxis, :, :, :, :]  # [1, 4, C, H, W]
#         x = to_var(torch.from_numpy(x))
#         predict_pose = model(x)
#         if i == 0:
#             pose_rel.extend(predict_pose.data.cpu().numpy())
#         else :
#             pose_rel.append(predict_pose[-1].data.cpu().numpy())
    pose_abl = relative2absolute(np.array(pose_rel))

    if args.phase == 'Test':
        np.savetxt((net_time_dir + '/{:02d}.txt'.format(seq)), pose_abl)
        plot_pose(seq=seq, label_dir=label_dir, save_dir=net_time_dir, pose_abl=pose_abl, args=args)
        print('Save pose and trajectory in {:s}'.format(net_time_dir))
    else:
        plot_pose(seq=seq, label_dir=label_dir, save_dir=model_nets_time, pose_abl=pose_abl, epoch=epoch, args=args)
        print('Save trajectory in {:s}'.format(model_nets_time))

def main():
    torch.set_default_tensor_type('torch.FloatTensor')
    model = Net()
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu)

    # Set weights
    print('\n========================================')
    print('Phase: {:s}\nNet architecture: {:s}'.format(args.phase, args.net_name))
    if args.resume == 'Yes' or args.phase == 'Test':
        print('rcnn restore from : {:s}'.format(restore_dir))
        xx = torch.load(restore_dir)
        model.load_state_dict(torch.load(restore_dir))

    elif args.resume == 'cnn':
        print('cnn restore from : {:s}'.format(restore_dir))
        pre_trained_dict = torch.load(restore_dir)
        model_dict = model.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}  # tick the useless dict
        model_dict.update(pre_trained_dict)  # update the dict
        model.load_state_dict(model_dict)  # load updated dict into the model

    else:
        print('Initialize from scratch')

    print('========================================')

    # Start training
    if args.phase == 'Train':
        model_nets_time, log_nets_time = folder_train(model_dir, log_dir, args)
        writer = SummaryWriter(log_nets_time)
        loss_func = nn.MSELoss()
        if args.resume == 'cnn':
            lstm_fc_params = list(map(id, model.module.lstm.parameters()))
            lstm_fc_params += list(map(id, model.module.fc_lstm_1.parameters()))
            lstm_fc_params += list(map(id, model.module.fc_lstm_2.parameters()))
            cnn_params = filter(lambda x: id(x) not in lstm_fc_params, model.parameters())
            for p in cnn_params:
                p.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_base)
            # for p in model.parameters():
            #     print(p.requires_grad)
        else:
            optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr_base * 0.1)

        data_t = KITTI_OF_LSTM(data_dir=data_dir, label_dir=label_dir, phase='Train', time_step=args.time_step)
        data_v = KITTI_OF_LSTM(data_dir=data_dir, label_dir=label_dir, phase='Validate', time_step=args.time_step)
        data_l_t = DataLoader(data_t, batch_size=args.batch_size, shuffle=True, num_workers=4)
        data_l_v = DataLoader(data_v, batch_size=args.batch_size, shuffle=False, num_workers=4)

        step_per_epoch = int(math.ceil(len(data_t) / data_l_v.batch_size))
        step_validate = int(math.floor(step_per_epoch / 3))  # 每个epoch验证3次
        min_loss_v = 1e10
        for epoch in np.arange(args.epoch_max):
            adjust_learning_rate(optimizer, epoch, args.lr_base, args.lr_decay_rate, args.epoch_lr_decay)

            # test a complete sequence and plot trajectory
            if epoch != 0 and epoch % 2 == 0:
                run_test(model, seq=7, model_nets_time=model_nets_time, epoch=epoch)
                run_test(model, seq=10, model_nets_time=model_nets_time, epoch=epoch)

            loss_list = []  # 记录每个epoch的loss
            loss1_list = []
            loss2_list = []
            loss_mean_train = 0
            for step, datas_t in enumerate(data_l_t):
                tic = time()
                step_global = epoch * step_per_epoch + step
                loss, loss1, loss2 = run_batch(datas_t, model=model, loss_func=loss_func, optimizer=optimizer,phase='Train')
                hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)
                loss_list.append(loss)
                loss1_list.append(loss1)
                loss2_list.append(loss2)
                loss_mean_train += float(loss)

                # display and add to tensor board
                if (step + 1) % 10 == 0:
                    display_t(hour_per_epoch, epoch, args, step, step_per_epoch, optimizer, loss, loss1,
                                    loss2, loss_list, loss1_list, loss2_list, writer, step_global)

                if (step + 1) % step_validate == 0:
                    batch_v = int(math.ceil(len(data_v) / data_l_v.batch_size))
                    loss_v, loss1_v, loss2_v = run_val(model, loss_func, data_l_v)
                    display_v(batch_v, loss_v, loss1_v, loss2_v, args, writer, step_global)
                    # if loss_v < min_loss_v:
                    #     min_loss_v = loss_v
                    #     print('Save model at ep {}, mean of valid loss: {}'.format(epoch + 1, loss_v))
                    #     print('\nSaving model_valid: {:s}/model-{:d}.pkl'.format(dir_model, epoch + 1))
                    #     torch.save(model.state_dict(), (dir_model + '/model-{:d}_valid.pkl'.format(epoch + 1)))

            if (epoch + 1) % 5 == 0:
                print('\nSaving model: {:s}/model-{:d}.pkl'.format(model_nets_time, epoch + 1))
                torch.save(model.state_dict(), (model_nets_time + '/model-{:d}.pkl'.format(epoch + 1)))


    else:
        net_time_dir = folder_test(test_dir, args)
        # import time
        # start = time.time()
        for seq in [4]:
            run_test(model, seq=seq, net_time_dir=net_time_dir)
        # end = time.time()
        # print("循环运行时间:%.2f秒" % (end - start))

if __name__ == '__main__':
    main()

