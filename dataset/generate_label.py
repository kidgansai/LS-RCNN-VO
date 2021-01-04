# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import mat
import math
import os

def isRotationMatrix(R):
    """ Checks if a matrix is a valid rotation matrix
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    """ calculates rotation matrix to euler angles
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])*180/math.pi
        y = math.atan2(-R[2,0], sy)*180/math.pi
        z = math.atan2(R[1,0], R[0,0])*180/math.pi
    else :
        x = math.atan2(-R[1,2], R[1,1])*180/math.pi
        y = math.atan2(-R[2,0], sy)*180/math.pi
        z = 0*180/math.pi

    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta):
    '''欧拉角转换为旋转矩阵'''
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

def generate_absolute2relative():
    dir_data = 'D:/sai/DeepVO/dataset/label/kitti-gt-6d/'
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)

    for Seq in range(11, 22):
        abl = np.loadtxt('D:/sai/DeepVO/dataset/label/kitti-gt-12d/{:02d}.txt'.format(Seq))
        rel = np.zeros((len(abl)-1, 6))
        for i in range(len(abl)-1):
            t1 = mat(np.row_stack((np.reshape(abl[i, :], (3, 4)), [0, 0, 0, 1])))
            t2 = mat(np.row_stack((np.reshape(abl[i+1, :], (3, 4)), [0, 0, 0, 1])))
            t12 = t1.I * t2
            rel[i, 0: 3] = [t12[0, 3], t12[1, 3], t12[2, 3]]
            Rot = t12[0:3,0:3]
            rel[i,3:6] = rotationMatrixToEulerAngles(Rot)           
        np.savetxt(dir_data + '{:02d}.txt'.format(Seq), rel)
        
generate_absolute2relative()

def validate_relative2absolute():
    for Seq in range(11, 22):
        gt = np.loadtxt('D:/sai/DeepVO/dataset/label/kitti-gt-12d/{:02d}.txt'.format(Seq))
        rel = np.loadtxt('D:/sai/DeepVO/dataset/label/kitti-gt-6d/{:02d}.txt'.format(Seq))
        t1 = mat(np.eye(4))
        abl = []
        abl.extend([np.array(t1[0: 3, :]).reshape([-1])])
        for i in range(len(rel)):
            x12 = rel[i, 0]
            y12 = rel[i, 1]
            z12 = rel[i, 2]
            theta = np.array([rel[i, 3] / 180 * math.pi,rel[i, 4] / 180 * math.pi,rel[i, 5] / 180 * math.pi])
            Rot  = eulerAnglesToRotationMatrix(theta)
            t12 = np.row_stack((np.column_stack((Rot, [[x12], [y12], [z12]])), [0, 0, 0, 1]))
            t2 = t1 * t12
            abl.extend([np.array(t2[0: 3, :]).reshape([-1])])
            t1 = t2
        abl = np.array(abl)
        # np.savetxt('{:d}.txt'.format(13), pose_absolute)
        plt.plot(gt[:, 3], gt[:, 11], '--', c='k', label='Ground Truth')
        plt.plot(abl[:, 3], abl[:, 11], c='r', label='Validate')
        plt.title('Sequence {:02d}'.format(Seq))
        plt.axis('equal')
        plt.legend()
        savedir = 'Sequence {:02d}.png'.format(Seq)
        plt.savefig(savedir)
        plt.close('all')
validate_relative2absolute()

def generate_absolute2relative_inverse():
        dir_data = 'D:/sai/label/kitti-inverse/'
        if not os.path.exists(dir_data):
            os.mkdir(dir_data)
        for Seq in range(11):
            abl = np.loadtxt('D:/sai/data_odometry_poses/dataset/poses/{:02d}.txt'.format(Seq))
            rel = np.zeros((len(abl) - 1, 6))
            for i in range(len(abl) - 1):
                t1 = mat(np.row_stack((np.reshape(abl[i, :], (3, 4)), [0, 0, 0, 1])))
                t2 = mat(np.row_stack((np.reshape(abl[i + 1, :], (3, 4)), [0, 0, 0, 1])))
                # 体现逆序的地方
                t12 = t2.I * t1
                rel[i, 0: 3] = [t12[0, 3], t12[1, 3], t12[2, 3]]
                Rot = t12[0:3, 0:3]
                rel[i, 3:6] = rotationMatrixToEulerAngles(Rot)
            np.savetxt(dir_data + '{:02d}.txt'.format(Seq), rel)

