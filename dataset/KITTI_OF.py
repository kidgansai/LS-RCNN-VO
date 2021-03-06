# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from func import read_of

class KITTI_OF(Dataset):

    def __init__(self, data_dir, label_dir,  phase=None, seq=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.phase = phase
        self.seq = seq
        if self.phase == 'Train' or self.phase == 'Validate':
            self.ol, self.label = self.train_data()
        else:
            self.ol, self.label = self.test_data()

    def train_data(self):
        ol = []
        label = []
        if self.phase == 'Train':
            for Seq in [3,10,1,4]:
            # for Seq in range(11):
                of_list = glob(self.data_dir + '/{:02d}/*.flo'.format(Seq))
                of_list.sort()
                ol.extend(of_list)
                load_label = np.loadtxt(self.label_dir + '/{:02d}.txt'.format(Seq))
                label.extend(load_label)
        else:
            validate_seq = 10
            of_list = glob(self.data_dir + '/{:02d}/*.flo'.format(validate_seq))
            of_list.sort()
            ol.extend(of_list)
            load_label = np.loadtxt(self.label_dir + '/{:02d}.txt'.format(validate_seq))
            label.extend(load_label)
        return ol, label

    def test_data(self):
        of_list = glob(self.data_dir + '/{:02d}/*.flo'.format(self.seq))
        of_list.sort()
        label = np.loadtxt(self.label_dir + '/{:02d}.txt'.format(self.seq))
        return of_list, label

    def __getitem__(self, index):
        datas = dict()
        of = read_of(self.ol[index])
        datas['of'] = np.transpose(of, [2, 0, 1])  # 6xHxW
        of_label = self.label[index]
        datas['label'] = of_label.astype(np.float32)
        return datas

    def __len__(self):
        return len(self.ol)

def main():

    # malaga数据集的地址和标签，实验时注意Malaga数据集没有0序列
    # data_dir = 'e:/sai/Malaga-brox/'
    # label_dir = 'D:/sai/LS-RCNN-VO/dataset/label/malaga-gt-6d'

    # KITTI数据集的地址和标签
    data_dir = 'e:/sai/kitti_flow_pwc'
    label_dir = 'D:/sai/LS-RCNN-VO/dataset/label/kitti-gt-6d'
    data_t = KITTI_OF(data_dir=data_dir, label_dir=label_dir, phase='Train')
    data_v = KITTI_OF(data_dir=data_dir, label_dir=label_dir, phase='Validate')
    data_l_t = DataLoader(data_t, batch_size=32, shuffle=False, num_workers=4)
    data_l_v = DataLoader(data_v, batch_size=32, shuffle=False, num_workers=4)
    n_batch = int(len(data_t.ol)//data_l_t.batch_size)
    for i_batch, data_batch in enumerate(data_l_t):
        print(i_batch, n_batch, data_batch['of'].size(), data_batch['of'].type(), data_batch['label'].size())
        if i_batch == 20:
            print('*' * 80)
            for i_batch_v, data_batch_v in enumerate(data_l_v):
                print(i_batch_v, data_batch_v['of'].size(), data_batch_v['of'].type(), data_batch_v['label'].size())
            print('*' * 80)

if __name__ == '__main__':
    main()
