import numpy as np
import matplotlib.pyplot as plt
import sys
import math

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

def calculate_sequence_speed_error(poses_gt,poses_result,step_size=10):
    # error_vetor
    errors = []
    # paramet
    step_size = step_size; # every second
    # pre-compute distances (from ground truth as reference)
    dist = trajectory_distances(poses_gt)
    # for all start positions do
    for  first_frame in range(0,poses_gt.shape[0]-step_size,step_size):
        last_frame = first_frame+step_size;
        # compute rotational and translational errors
        pose_delta_gt     = line2matrix(poses_gt[first_frame]).I*line2matrix(poses_gt[last_frame])
        pose_delta_result = line2matrix(poses_result[first_frame]).I*line2matrix(poses_result[last_frame])
        pose_error        = pose_delta_result.I*pose_delta_gt;
        r_err = rotation_error(pose_error);
        t_err = translation_error(pose_error);
        length =dist[last_frame]-dist[first_frame]
        speed = length*3.6;
        # write to file
        error = [first_frame,r_err/length,t_err/length,length,speed]
        errors.append(error)
        # return error vector
    return errors

def calculate_ave_speed_errors(errors):
    speed = [x[-1] for x in errors]
    speed = np.array(speed)
    speed_max = speed.max()
    speed_min = speed.min()
    speed_max = math.ceil(speed_max/10)
    speed_min = int(speed_min/10)
    lengths = []
    for i in range(speed_min,speed_max):
        lengths.append((i+1)*10)
    # print(lengths)

    rot_errors=[]
    tra_errors=[]
    for length in lengths:
        rot_error_each_speed =[]
        tra_error_each_speed =[]
        for error in errors:
            if length-error[4]<10 and length-error[4]>0 :
                rot_error_each_speed.append(error[1])
                tra_error_each_speed.append(error[2])

        rot_errors.append(sum(rot_error_each_speed)/len(rot_error_each_speed))
        tra_errors.append(sum(tra_error_each_speed)/len(tra_error_each_speed))
    return rot_errors,tra_errors,lengths

def plot_figure(x,y,index,fmt_list, colors, legend_list, dir_save,Seq):
    for i in range(len(x)):
        plt.plot(x[i], y[i], linestyle =fmt_list[i+1],color = colors[i+1],marker='o',lw=1.5, label=legend_list[i+1]) #mec -- 折点外边颜色, mfc --折点实心颜色
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if index == 0:
        plt.xlabel("Path Length[m]", fontsize=12)  # X轴标签
        plt.ylabel("Translation Error[%]", fontsize=12)  # Y轴标签
        # plt.ylim((0,40))
    if index == 1:
        plt.xlabel("Path Length[m]", fontsize=12)  # X轴标签
        plt.ylabel("Rotation Error[deg/m]", fontsize=12)  # Y轴标签
        # plt.ylim((0, 0.1))
    if index == 2:
        plt.xlabel("Speed[km/h]", fontsize=12)  # X轴标签
        plt.ylabel("Translation Error[%]", fontsize=12)  # Y轴标签
        # plt.ylim((0, 40))
    if index == 3:
        plt.xlabel("Speed[km/h]", fontsize=12)  # X轴标签
        plt.ylabel("Rotation Error[deg/m]", fontsize=12)  # Y轴标签
        # plt.ylim((0, 0.1))
    plt.title("Sequence {:02d}".format(Seq))
    plt.grid(True)
    plt.legend()  # 让图例生效
    plt.savefig(dir_save+'.png')
    plt.close('all')

def plot_evaluation(file_list, fmt_list, colors, legend_list, dir_save):
    # ave_errors = []
    for Seq in range(11):
        lengths_length = []
        tra_length = []
        rot_length = []
        lengths_speed = []
        tras_speed = []
        rots_speed = []
        for i in range(len(file_list)-1):
            path_pre = file_list[i+1] + '%02d' % Seq + '.txt'
            path_gt = file_list[0] + '%02d' % Seq + '.txt'
            ground_truth_data = get_poses(path_gt)
            predict_pose_data = get_poses(path_pre)
            tra_lengths = trajectory_distances(ground_truth_data)
            # print(tra_lengths[-1])
            if tra_lengths[-1] >= 800:
                lengths = [100, 200, 300, 400, 500, 600, 700, 800]
            else:
                lengths = []
                for i in range(int(tra_lengths[-1] / 100)):
                    lengths.append((i + 1) * 100)
            # print(lengths)
            errors = calculate_sequence_error(ground_truth_data, predict_pose_data, lengths)
            rot, tra = calculate_ave_errors(errors, lengths)
            # print(np.mean(rot)*100)
            # print(np.mean(tra)*100)
            # ave_errors.append([np.mean(tra), np.mean(rot)])
            lengths_length.append(lengths)
            tra_length.append(tra)
            rot_length.append(rot)

            errors_speed = calculate_sequence_speed_error(ground_truth_data, predict_pose_data)
            rot_speed, tra_speed, length_speed = calculate_ave_speed_errors(errors_speed)
            lengths_speed.append(length_speed)
            tras_speed.append(tra_speed)
            rots_speed.append(rot_speed)

        plot_figure(lengths_length, tra_length, 0, fmt_list, colors, legend_list, dir_save + 'Translations_error_path_length' + '%02d' % Seq, Seq)
        plot_figure(lengths_length, rot_length, 1, fmt_list, colors, legend_list, dir_save + 'Rolations_error_path_length' + '%02d' % Seq, Seq)
        plot_figure(lengths_speed, tras_speed, 2, fmt_list, colors, legend_list, dir_save + 'Translations_error_speed' + '%02d' % Seq, Seq)
        plot_figure(lengths_speed, rots_speed, 3, fmt_list, colors, legend_list,dir_save + 'Rolations_error_speed' + '%02d' % Seq, Seq)
    # print(ave_errors)
    # for i in range(len(ave_errors)):
    #     fh = open("D:/test/ave_errors.txt", "a")
    #     fh.write("%f %f\n" % (ave_errors[i][0], ave_errors[i][1]))
    #     fh.close()


# 测试
file_list = ['D:/sai/data_odometry_poses/dataset/poses/',
             'C:/Users/DELL/Desktop/cnn-vo/20190415_model-100/',
             'C:/Users/DELL/Desktop/cnn-vo/20190415_model-110/',
             'C:/Users/DELL/Desktop/cnn-vo/20190415_model-115/',
             'C:/Users/DELL/Desktop/cnn-vo/20190415_model-90/']
fmt_list = ['--', '-', '-', '-', '-']
colors = ['k', 'c', 'r', 'g', 'b']
legend_list = ['Ground truth', 'VISO2-M', 'VISO2-S', 'CNN-VO', 'CNN-VO-cons']
dir_save = 'D:/test/'
plot_evaluation(file_list,fmt_list,colors,legend_list,dir_save)


def plot_trajectory(sequence, file_list, fmt_list, colors, legend_list, dir_save):
    x = []
    z = []
    for i in range(len(file_list)):
        path = file_list[i] + '%02d' % sequence + '.txt'
        pose = get_poses(path)
        x_ = []
        z_ = []
        for j in range(len(pose)):
            x_.append(pose[j][3])
            z_.append(pose[j][11])
        x.append(x_)
        z.append(z_)
    for i in range(len(x)):
        plt.plot(x[i], z[i], linestyle =fmt_list[i],color = colors[i], label=legend_list[i])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Sequence {:02d}".format(sequence))
    plt.grid(True)
    plt.legend()  # 让图例生效
    plt.savefig(dir_save + '%02d' % sequence+ '.png')
    plt.close('all')
# 测试
# file_list = ['D:/sai/data_odometry_poses/dataset/poses/',
#              'C:/Users/DELL/Desktop/cnn-vo/20190415_model-100/',
#              'C:/Users/DELL/Desktop/cnn-vo/20190415_model-110/',
#              'C:/Users/DELL/Desktop/cnn-vo/20190415_model-115/',
#              'C:/Users/DELL/Desktop/cnn-vo/20190415_model-90/']
# fmt_list = ['--', '-', '-', '-', '-']
# colors = ['k', 'c', 'r', 'g', 'b']
# legend_list = ['Ground truth', 'VISO2-M', 'VISO2-S', 'CNN-VO', 'CNN-VO-cons']
# dir_save = 'D:/test/'
# plot_trajectory(1,file_list,fmt_list,colors,legend_list,dir_save)
