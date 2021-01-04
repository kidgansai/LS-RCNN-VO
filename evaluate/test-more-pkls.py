# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.utils.data import DataLoader
import re
import argparse
from glob import glob
from func import *
from net.cnn import Net
from dataset.KITTI_OF import KITTI_OF

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='Test', help='[Train / Test]')
parser.add_argument('--resume', default='Yes', help='[Yes / No] for cnn, [cnn / lstm / No] for cnn-lstm')
# 权重恢复的路径
parser.add_argument('--net_restore', default='ae-vo', help='Restore net name')
parser.add_argument('--dir_restore', default='20201010', help='Restore time')
parser.add_argument('--model_restore', default='model-validate', help='Restore model-id')
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

def run_test(model, seq, net_time_dir=None):
    print('\nTest sequence {:02d} >>>'.format(seq))
    data = KITTI_OF(data_dir=data_dir, label_dir=label_dir, phase='Test', seq=seq)
    data_l = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pose_rel = []
    for _, data_batch in enumerate(tqdm(data_l)):
        pose_pre = run_batch(datas=data_batch, model=model, phase='Test')
        pose_rel.extend(pose_pre.cpu().numpy())
    pose_abl = relative2absolute(pose_rel)
    np.savetxt((net_time_dir + '/{:02d}.txt'.format(seq)), pose_abl)
    plot_pose(seq=seq, label_dir=label_dir, save_dir=net_time_dir, pose_abl=pose_abl, args=args)
    print('Save pose and trajectory in {:s}'.format(net_time_dir))

def main():
    torch.set_default_tensor_type('torch.FloatTensor')
    model = Net()
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu)
    # Set weights
    print('\n========================================')
    print('Phase: {:s}\nNet architecture: {:s}'.format(args.phase, args.net_name))
    dir_net = 'test/' + args.net_restore
    if not os.path.exists(dir_net):
        os.mkdir(dir_net)
    pkl_list = glob(model_dir+ '/' + args.net_restore + '/' + args.dir_restore+'/*.pkl')
    pkl_list.sort()
    if not os.path.exists(dir_net+'/' + args.dir_restore):
        os.mkdir(dir_net+'/' + args.dir_restore)
    for i in range(len(pkl_list)):
        if args.resume == 'Yes' or args.phase == 'Test':
            print('Restore from CNN: {:s}'.format(pkl_list[i]))
            model.load_state_dict(torch.load(pkl_list[i]))
        print('========================================')
        if os.path.splitext(pkl_list[i])[1] == '.pkl':
            fileName = os.path.splitext(pkl_list[i])[0]
            fileName = fileName.split("/")[-1]
            # print(fileName)
            test_path = dir_net+'/'+fileName
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        for seq in [9,10]:
        # for seq in range(11):
            run_test(model, seq=seq, net_time_dir=test_path)

if __name__ == '__main__':
    main()
