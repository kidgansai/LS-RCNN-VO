import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import mat
import math
import os

def get_poses(path):
    with open(path) as f:
        poses = np.array([[float(x) for x in line.split()] for line in f])
    return poses

def trajectory_distances(poses):
    distances = []
    distances.append(0)
    for i in range(1,poses.shape[0]):
        p1 = poses[i-1]
        p2 = poses[i]
        delta = p1[3:12:4] - p2[3:12:4]
        distances.append(distances[i-1]+np.linalg.norm(delta))
    return distances

def last_frame_from_segment_length(dist,first_frame,length):
    for i in range(first_frame,len(dist)):
        if dist[i]>dist[first_frame]+length:
            return i
    return -1

def rotation_error(pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5*(a+b+c-1)
    rot_error = np.arccos(max(min(d,1.0),-1.0))
    return rot_error

def translation_error(pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx*dx+dy*dy+dz*dz)

def line2matrix(pose_line):
    pose_line = np.array(pose_line)
    pose_m = np.matrix(np.eye(4))
    pose_m[0:3,:] = pose_line.reshape(3,4)
    return pose_m
def calculate_sequence_error(poses_gt,poses_result,lengths):
    # error_vetor
    errors = []

    # paramet
    step_size = 10; # every second
    lengths   = lengths
    num_lengths = len(lengths)

    # pre-compute distances (from ground truth as reference)
    dist = trajectory_distances(poses_gt)
    # for all start positions do
    for  first_frame in range(0,poses_gt.shape[0],step_size):
    # for all segment lengths do
        for i in range(0,num_lengths):
            #  current length
            length = lengths[i];

            # compute last frame
            last_frame = last_frame_from_segment_length(dist,first_frame,length);
            # continue, if sequence not long enough
            if (last_frame==-1):
                continue;

            # compute rotational and translational errors
            pose_delta_gt     = line2matrix(poses_gt[first_frame]).I*line2matrix(poses_gt[last_frame])
            pose_delta_result = line2matrix(poses_result[first_frame]).I*line2matrix(poses_result[last_frame])
            pose_error        = pose_delta_result.I*pose_delta_gt;
            r_err = rotation_error(pose_error);
            t_err = translation_error(pose_error);

            # compute speed
            num_frames = (float)(last_frame-first_frame+1);
            speed = length/(0.1*num_frames);

            # write to file
            error = [first_frame,r_err/length*100,t_err/length*100,length,speed]
            errors.append(error)
            # return error vector
    return errors
def calculate_ave_errors(errors,lengths):
    lengths = lengths
    rot_errors=[]
    tra_errors=[]
    for length in lengths:
        rot_error_each_length =[]
        tra_error_each_length =[]
        for error in errors:
            if abs(error[3]-length)<1:
                rot_error_each_length.append(error[1])
                tra_error_each_length.append(error[2])

        rot_errors.append(sum(rot_error_each_length)/len(rot_error_each_length))
        tra_errors.append(sum(tra_error_each_length)/len(tra_error_each_length))
    return rot_errors,tra_errors

def file_name(file_dir):
    sub_dirs = []
    for root, dirs, files in os.walk(file_dir):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件
        sub_dirs.append(dirs)
        # print(subdirs_[0][2])
    return sub_dirs

def  main():

    subdirs = file_name('D:/sai/pytorch-cnnvo/test/of-vo/20200702')
    for i in range(len(subdirs[0])):
        path = 'D:/sai/pytorch-cnnvo/test/of-vo/20200702'+'/' + subdirs[0][i] + '/'
        ave_errors = []
        for Seq in [9]:
            path_pre = path+'%02d' % Seq +'.txt'
            path_gt = 'D:/sai/data_odometry_poses/dataset/poses/'+'%02d' % Seq +'.txt'
            ground_truth_data  = get_poses(path_gt)
            predict_pose_data = get_poses(path_pre)
            tra_lengths = trajectory_distances(ground_truth_data)
            if tra_lengths[-1]>=800:
                lengths = [100,200,300,400,500,600,700,800]
            else:
                lengths = []
                for i in range(int(tra_lengths[-1]/100)):
                    lengths.append((i+1)*100)
            errors = calculate_sequence_error(ground_truth_data,predict_pose_data,lengths)
            rot,tra = calculate_ave_errors(errors,lengths)

            ave_errors.append([np.mean(tra),np.mean(rot)])
        print("="*50)
        print(path)
        print(ave_errors)

if __name__ == "__main__":
    main()


