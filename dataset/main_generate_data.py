# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Gerry
Email: 1406797728@qq.com
Create Time: 2019-5-25 10:55:13
"""
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from numpy import mat
import math
import os


def generate_xyz_euler_absolute():
    for i in range(11):
        t_12d = np.loadtxt('T-12d/{:d}.txt'.format(i))
        ret = np.zeros((len(t_12d), 6))
        for j in range(len(t_12d)):
            t1 = t_12d[j, :]
            ret[j, 0: 3] = [t1[3], t1[7], t1[11]]

            # rotation matrix to ypr
            pitch = math.atan2(-t1[8], np.sqrt(t1[0] * t1[0] + t1[4] * t1[4])) / 3.14 * 180
            if pitch == -90:
                yaw = math.atan2(-t1[9], -t1[2]) / 3.14 * 180
                roll = 0
            elif pitch == 90:
                yaw = math.atan2(t1[6], t1[2]) / 3.14 * 180
                roll = 0
            else:
                yaw = math.atan2(t1[4], t1[0]) / 3.14 * 180
                roll = math.atan2(t1[9], t1[10]) / 3.14 * 180
            ret[j, 3] = yaw
            ret[j, 4] = pitch
            ret[j, 5] = roll

        np.savetxt('xyz-euler-absolute/{:d}.txt'.format(i), ret)


def generate_xyz_euler_relative(it):
    dir_data = 'xyz-euler-relative-interval{:d}/'.format(it)
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)

    for i in tqdm(range(11)):
        t_12d = np.loadtxt('ground-truth/{:02d}.txt'.format(i))
        print(t_12d.shape)
        ret = np.zeros((len(t_12d) - it - 1, 6))
        print(ret.shape)
        for j in range(len(t_12d) - it - 1):
            t1 = mat(np.row_stack((np.reshape(t_12d[j, :], (3, 4)), [0, 0, 0, 1])))  # 转换为旋转矩阵
            t2 = mat(np.row_stack((np.reshape(t_12d[j + it + 1, :], (3, 4)), [0, 0, 0, 1])))
            # print("t1", t1)
            # print("t2", t2)
            t12 = t1.I * t2  # 求逆矩阵
            ret[j, 0: 3] = [t12[0, 3], t12[1, 3], t12[2, 3]]

            # rotation matrix to ypr
            theta_y = math.atan2(-t12[2, 0], np.sqrt(t12[0, 0] * t12[0, 0] + t12[1, 0] * t12[1, 0])) / 3.14 * 180
            if theta_y == -90:
                theta_x = 0
                theta_z = math.atan2(-t12[2, 0], -t12[0, 2]) / 3.14 * 180

            elif theta_y == 90:
                theta_x = 0
                theta_z = math.atan2(t12[1, 2], t12[0, 2]) / 3.14 * 180
            else:
                theta_x = math.atan2(t12[2, 1], t12[2, 2]) / 3.14 * 180
                theta_z = math.atan2(t12[1, 0], t12[0, 0]) / 3.14 * 180
            ret[j, 3] = theta_x
            ret[j, 4] = theta_y
            ret[j, 5] = theta_z

        np.savetxt(dir_data + '{:d}.txt'.format(i), ret)


def validate_xyz_euler_relative(seq):
    pose_gt = np.loadtxt('ground-truth/{:02d}.txt'.format(seq))
    xyz_euler = np.loadtxt('xyz-euler-relative-interval0/{:d}.txt'.format(seq))
    t1 = mat(np.eye(4))
    pose_absolute = []  # 12-d
    pose_absolute.extend([np.array(t1[0: 3, :]).reshape([-1])])
    for i in tqdm(range(len(xyz_euler))):
        x12 = xyz_euler[i, 0]
        y12 = xyz_euler[i, 1]
        z12 = xyz_euler[i, 2]
        theta1 = xyz_euler[i, 3] / 180 * np.pi
        theta2 = xyz_euler[i, 4] / 180 * np.pi
        theta3 = xyz_euler[i, 5] / 180 * np.pi
        tx = mat([[1, 0, 0], [0, math.cos(theta1), -math.sin(theta1)], [0, math.sin(theta1), math.cos(theta1)]])
        ty = mat([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
        tz = mat([[math.cos(theta3), -math.sin(theta3), 0], [math.sin(theta3), math.cos(theta3), 0], [0, 0, 1]])
        tr = tz * ty * tx
        t12 = np.row_stack((np.column_stack((tr, [[x12], [y12], [z12]])), [0, 0, 0, 1]))
        t2 = t1 * t12
        pose_absolute.extend([np.array(t2[0: 3, :]).reshape([-1])])
        t1 = t2

    pose_absolute = np.array(pose_absolute)
    plt.plot(pose_gt[:, 3], pose_gt[:, 11], '--', c='k', lw=1.5, label='Ground truth')
    plt.plot(pose_absolute[:, 3], pose_absolute[:, 11], '-', c='r', lw=1.5, label='Test')
    plt.title('Sequence {:02d}'.format(seq))
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def generate_xyz_euler_relative_reverse(it):
    dir_data = 'xyz-euler-relative-reverse-interval{:d}/'.format(it)
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)
    for i in tqdm(range(11)):
        gt = np.loadtxt('ground-truth/{:02d}.txt'.format(i))
        ret = np.zeros((len(gt) - it - 1, 6))
        for j in range(len(gt) - it - 1):
            t1 = mat(np.row_stack((np.reshape(gt[j, :], (3, 4)), [0, 0, 0, 1])))
            t2 = mat(np.row_stack((np.reshape(gt[j + it + 1, :], (3, 4)), [0, 0, 0, 1])))
            t12 = t2.I * t1
            ret[j, 0: 3] = [t12[0, 3], t12[1, 3], t12[2, 3]]

            # rotation matrix to ypr
            theta_y = math.atan2(-t12[2, 0], np.sqrt(t12[0, 0] * t12[0, 0] + t12[1, 0] * t12[1, 0])) / 3.14 * 180
            if theta_y == -90:
                theta_x = 0
                theta_z = math.atan2(-t12[2, 0], -t12[0, 2]) / 3.14 * 180

            elif theta_y == 90:
                theta_x = 0
                theta_z = math.atan2(t12[1, 2], t12[0, 2]) / 3.14 * 180
            else:
                theta_x = math.atan2(t12[2, 1], t12[2, 2]) / 3.14 * 180
                theta_z = math.atan2(t12[1, 0], t12[0, 0]) / 3.14 * 180
            ret[j, 3] = theta_x
            ret[j, 4] = theta_y
            ret[j, 5] = theta_z

        np.savetxt(dir_data + '{:d}.txt'.format(i), ret)


def generate_xzp_absolute():
    for i in range(11):
        t_12d = np.loadtxt('T-12d/{:d}.txt'.format(i))
        xzp_absolute = np.zeros((len(t_12d), 3))
        for j in range(len(t_12d)):
            t1 = t_12d[j, :]
            xzp_absolute[j, 0: 2] = [t1[3], t1[11]]

            # rotation matrix to ypr
            pitch = math.atan2(-t1[8], np.sqrt(t1[0] * t1[0] + t1[4] * t1[4])) / 3.14 * 180
            xzp_absolute[j, 2] = pitch

        np.savetxt('xzp-absolute/{:d}.txt'.format(i), xzp_absolute)


def generate_xzp_relative():
    for i in range(11):
        xzp_absolute = np.loadtxt('xzp-absolute/{:d}.txt'.format(i))
        xzp_relative = np.zeros((len(xzp_absolute) - 1, 3))
        x1 = xzp_absolute[0, 0]
        z1 = xzp_absolute[0, 1]
        p1 = xzp_absolute[0, 2] / 180 * 3.14
        t1 = mat([[math.cos(p1), -math.sin(p1), x1], [math.sin(p1), math.cos(p1), z1], [0, 0, 1]])
        for j in range(len(xzp_absolute) - 1):
            x2 = xzp_absolute[j + 1, 0]
            z2 = xzp_absolute[j + 1, 1]
            p2 = xzp_absolute[j + 1, 2] / 180 * 3.14
            t2 = mat([[math.cos(p2), -math.sin(p2), x2], [math.sin(p2), math.cos(p2), z2], [0, 0, 1]])
            t12 = t1.I * t2
            xzp_relative[j, 0: 2] = [t12[0, 2], t12[1, 2]]
            pitch = math.atan2(t12[1, 0], t12[0, 0]) / 3.14 * 180
            xzp_relative[j, 2] = pitch
            t1 = t2
        np.savetxt('xzp-relative/{:d}.txt'.format(i), xzp_relative)


def validate_xzp_relative(seq):
    xzp = np.loadtxt('xzp-relative/{:d}.txt'.format(seq))
    t1 = mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in tqdm(range(len(xzp))):
        x12 = xzp[i, 0]
        z12 = xzp[i, 1]
        p12 = xzp[i, 2] / 180 * 3.14
        t12 = mat([[math.cos(p12), -math.sin(p12), x12], [math.sin(p12), math.cos(p12), z12], [0, 0, 1]])
        t2 = t1 * t12
        plt.plot((t1[0, 2], t2[0, 2]), (t1[1, 2], t2[1, 2]), '-b', label='Validation')
        t1 = t2

    plt.axis('equal')
    plt.grid('on')
    plt.title('Sequence {:d}'.format(seq))
    plt.show()


def generate_xz_new_absolute():
    for i in range(11):
        t_12d = np.loadtxt('T-12d/{:d}.txt'.format(i))
        xzp_absolute = np.zeros((len(t_12d), 2))
        for j in range(len(t_12d)):
            xzp_absolute[j, :] = [t_12d[j, 3], t_12d[j, 11]]
        np.savetxt('xz-new-absolute/{:d}.txt'.format(i), xzp_absolute)


def validate_xz_new_absolute(seq):
    vec = np.loadtxt('xz-new-absolute/{:d}.txt'.format(seq))
    for i in range(1, len(vec), 3):
        plt.plot((vec[i - 1, 0], vec[i, 0]), (vec[i - 1, 1], vec[i, 1]), '-b', label='Ground truth')

    plt.axis('equal')
    plt.grid('on')
    plt.title('Sequence {:d}'.format(seq))
    plt.show()


def generate_xz_new_relative():
    for i in range(11):
        t_12d = np.loadtxt('T-12d/{:d}.txt'.format(i))
        xz_new = np.zeros((len(t_12d) - 1, 2))
        for j in range(len(t_12d) - 1):
            xz_new[j, :] = [t_12d[j + 1, 3] - t_12d[j, 3], t_12d[j + 1, 11] - t_12d[j, 11]]
        np.savetxt('xz-new-relative/{:d}'.format(i), xz_new)


def validate_xz_new_relative(seq):
    xz = np.loadtxt('xz-new-relative/{:d}'.format(seq))
    t1 = [0, 0]
    for i in range(200):  # range(len(xz)):
        t2 = t1 + xz[i]
        plt.plot((t1[0], t2[0]), (t1[1], t2[1]), '-b', label='Ground truth')
        t1 = t2
    plt.axis('equal')
    plt.grid('on')
    plt.title('Sequence {:d}'.format(seq))
    plt.show()


def plot_pose():
    for i in range(11):
        pose_gt = np.loadtxt('ground-truth/{:02d}.txt'.format(i))
        print(pose_gt.shape)
        # plt.subplot(341 + (i % 4))
        plt.plot(pose_gt[:, 3], pose_gt[:, 11], '--', c='b', lw=1.5)
        plt.title('Sequence {:02d}'.format(i))
        plt.grid(True)
        # plt.legend()
        plt.show()


import copy


def plot():
    pose_gt = np.loadtxt('ground-truth/{:02d}.txt'.format(11))
    print(pose_gt.shape)
    # print(len(pose_gt))
    X = copy.deepcopy(pose_gt[:, 0])
    Y = copy.deepcopy(pose_gt[:, 3])
    # for i in range(len(pose_gt)):
    #     X[i] += 0.001 * i
    #     Y[i] -= 0.01 * i

    # plt.plot(pose_gt[:, 3], pose_gt[:, 11], '--', c='b', lw=1.5)
    plt.plot(X, Y, '-', c='r', lw=2)
    plt.title('Sequence {:02d}'.format(0))
    plt.grid(True)
    # plt.legend()
    plt.show()


def plot_historm():
    x_data = []
    y_data = []
    z_data = []
    x1_data = []
    y1_data = []
    z1_data = []
    for i in range(11):
        data = np.loadtxt('xyz-euler-relative-interval0/{:d}.txt'.format(i))
        x_data.extend(data[:, 0])
        y_data.extend(data[:, 1])
        z_data.extend(data[:, 2])
        x1_data.extend(data[:, 3])
        y1_data.extend(data[:, 4])
        z1_data.extend(data[:, 5])
    plt.figure(0)
    plt.hist(x_data, bins=50, color='b', normed=True)
    plt.title("Histogram of X")
    plt.show()

    plt.figure(1)
    plt.hist(y_data, bins=50, color='b', normed=True)
    plt.title("Histogram of Y")
    plt.show()
    plt.figure(2)
    plt.hist(z_data, bins=50, color='b', normed=True)
    plt.title("Histogram of Z")
    plt.show()

    plt.figure(3)
    plt.hist(x1_data, bins=50, color='b', normed=True)
    plt.title("Histogram of ψ")
    plt.show()

    plt.figure(4)
    plt.hist(y1_data, bins=50, color='b', normed=True)
    plt.title("Histogram of χ")
    plt.show()

    plt.figure(5)
    plt.hist(z1_data, bins=50, color='b', normed=True)
    plt.title("Histogram of φ")
    plt.show()


if __name__ == '__main__':
    # generate_xyz_euler_relative(0)
    # generate_xyz_euler_relative(1)
    # generate_xyz_euler_relative(2)
    # generate_xyz_euler_relative(3)
    # generate_xyz_euler_relative_reverse(0)
    # generate_xyz_euler_relative_reverse(1)
    # generate_xyz_euler_relative_reverse(2)
    # for i in range(11):
    #     generate_xyz_euler_relative(i)
    #     generate_xyz_euler_relative_reverse(i)
    #     validate_xyz_euler_relative(i)
    plot()
    # plot_historm()
