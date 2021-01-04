# -*- coding: utf-8 -*-
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from func import read_of
class KITTI_OF_LSTM(Dataset):

    def __init__(self, data_dir, label_dir, time_step=4,  phase=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.phase = phase
        self.time_step = time_step
        self.ol, self.start_index, self.label = self.load_data()

    def load_data(self):
        ol = []
        start_index = []
        label = []
        if self.phase == 'Train':
            count = 0
            for Seq in [4]:
            # for Seq in range(11):
                of_list = glob(self.data_dir + '/{:02d}/*.flo'.format(Seq))
                of_list.sort()
                ol.extend(of_list)
                load_label = np.loadtxt(self.label_dir + '/{:02d}.txt'.format(Seq))
                for i in np.arange(len(of_list)):
                    label_datas = np.zeros((self.time_step, 6))  # [4, 6]
                    if (i < (len(of_list)-self.time_step+1)):
                        start_index.append(count)
                        label_datas = load_label[i: i + self.time_step]
                    label.append(label_datas)
                    count += 1
        else:
            validate_seq = 4
            of_list = glob(self.data_dir + '/{:02d}/*.flo'.format(validate_seq))
            of_list.sort()
            ol.extend(of_list)
            load_label = np.loadtxt(self.label_dir + '/{:02d}.txt'.format(validate_seq))
            for i in np.arange(len(of_list)):
                label_datas = np.zeros((self.time_step, 6))  # [10, 6]
                if i < (len(of_list) - self.time_step+1):
                    start_index.append(i)
                    label_datas = load_label[i: i + self.time_step]
                label.append(label_datas)

        return ol, start_index, label

    def __getitem__(self, index):
        datas = dict()
        index = self.start_index[index]
        # print(index)
        of_list = []
        for of_path in self.ol[index: index + self.time_step]:
            of = read_of(of_path)
            of_list.append(of.astype(np.float32))
        datas['of'] = of_list
        datas['label'] = np.array(self.label[index]).astype(np.float32)

        datas['of'] = np.stack(datas['of'], 0)  # list ==> TxHxWxC 增加维度
        datas['of'] = np.transpose(datas['of'], [0, 3, 1, 2])  # TxHxWx2 ==> TxCxHxW

        return datas

    def __len__(self):
        return len(self.start_index)

def main():

    # malaga数据集的地址和标签，实验时注意Malaga数据集没有0序列
    # data_dir = 'e:/sai/Malaga-brox/'
    # label_dir = 'D:/sai/LS-RCNN-VO/dataset/label/malaga-gt-6d'

    # KITTI数据集的地址和标签
    data_dir =  'e:/sai/kitti_flow_pwc/'
    label_dir = 'D:/sai/DeepVO/dataset/label'
    data_t = KITTI_OF_LSTM(data_dir=data_dir, label_dir=label_dir,time_step=4, phase='Train')
    data_l_t = DataLoader(data_t, batch_size=4, shuffle=True, num_workers=4)
    print('ts {},  Total datas {}, bs {}'.format(data_t.time_step,data_t.__len__(), data_l_t.batch_size))
    n_batch = int(len(data_t.ol)//data_l_t.batch_size)
    for i_batch, data_batch in enumerate(data_l_t):
        print(i_batch, n_batch, data_batch['of'].size(), data_batch['of'].type(), data_batch['label'].size())
if __name__ == '__main__':
    main()
