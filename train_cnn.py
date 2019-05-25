#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Gerry
Email: 1406797728@qq.com
Create Time: 2019-5-25 10:55:13
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import re
import os
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from glob import glob
from tensorboardX import SummaryWriter
from Pytorch.DeepVO.utils.post_process import cal_absolute_from_relative, plot_from_pose
from Pytorch.DeepVO.utils.misc import to_var, adjust_learning_rate, pre_create_file_train, pre_create_file_test, \
    display_loss_tb, display_loss_tb_val

parser = argparse.ArgumentParser()
parser.add_argument('--server', default=6499, type=int, help='[6099 / 6199 / 6499]')
parser.add_argument('--net_architecture', default='cnn', help='[cnn / cnn-tb / cnn-iks / cnn-lstm]')
parser.add_argument("--samples", default='i0', help='samples for train')
parser.add_argument('--phase', default='Train', help='[Train / Test]')
parser.add_argument('--resume', default='No', help='[Yes / No] for cnn, [cnn / lstm / No] for cnn-lstm')

# 模型载入的参数
parser.add_argument('--net_restore', default='cnn-vo', help='Restore net name')
parser.add_argument('--dir_restore', default='20190101', help='Restore file name')
parser.add_argument('--model_restore', default='model-200', help='Restore model-id')

parser.add_argument('--net_name', default='cnn-vo', help='[cnn-vo / cnn-vo-cons / cnn-lstm-vo / cnn-lstm-vo-cons]')
parser.add_argument('--dir0', default=20190418, help='Name it with date, such as 20180102')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--epoch_max', default=100, type=int, help='Max epoch')
parser.add_argument('--epoch_test', default=10, type=int, help='Test epoch during train process')
parser.add_argument('--epoch_save', default=10, type=int, help='Max epoch number')
parser.add_argument('--lr_base', default=1e-4, type=float, help='Base learning rate')
parser.add_argument('--lr_decay_rate', default=0.316, type=float, help='Decay rate of lr')
parser.add_argument('--epoch_lr_decay', default=30, type=int, help='Every # epoch, lr decay lr_decay_rate')
parser.add_argument('--beta', default=10, type=int, help='loss = loss_t + beta * loss_r')

# lstm 参数
parser.add_argument('--img_pairs', default=10, type=int, help='Image pairs')
parser.add_argument('--si', default=3, type=int, help='Start interval')
parser.add_argument('--num_layer', default=2, type=int, help='Lstm layer number')
parser.add_argument('--hidden_size', default=1024, type=int, help='Lstm hidden units')

parser.add_argument("--gpu", default='0', help='GPU id list')
parser.add_argument("--workers", default=4, type=int, help='Workers number')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见的gpu的列表，例如：'2,3,4'
gpu_list = re.split('[, ]', args.gpu)  # 提取出列表中gpu的id
args.gpu = range(len(list(filter(None, gpu_list))))  # 传给PyTorch中多gpu并行的列表

if args.server == 6099:
    dir_data = str('E:\data\dataset\sequences')
    dir_label = 'dataset'
    model_dir = 'model'
    log_dir = 'log'
elif args.server == 6199:
    dir_data = 'E:\data\dataset\sequences'
    dir_label = 'dataset'
    model_dir = 'model'
    log_dir = 'log'
elif args.server == 6499:
    dir_data = 'E:\data\dataset\sequences'
    dir_label = 'dataset'
    model_dir = 'model'
    log_dir = 'log'
else:
    raise Exception('Must give the right server id!')

dir_restore = model_dir + '/' + args.net_restore + '/' + args.dir_restore + '/' + args.model_restore + '.pkl'

if args.net_architecture == 'cnn':
    from Pytorch.DeepVO.net.cnn import Net
    from Pytorch.DeepVO.dataset.kitti import KITTIDataSet
elif args.net_architecture == 'cnn-sc':
    from Pytorch.DeepVO.net.cnn_seperate_conv import Net
    from Pytorch.DeepVO.dataset.kitti import KITTIDataSet
elif args.net_architecture == 'cnn-sc1':
    from Pytorch.DeepVO.net.cnn_seperate_conv_1 import Net
    from Pytorch.DeepVO.dataset.kitti import KITTIDataSet
elif args.net_architecture == 'cnn-tb':
    from Pytorch.DeepVO.net.cnn_tb import Net
    from Pytorch.DeepVO.dataset.kitti import KITTIDataSet
elif args.net_architecture == 'cnn-iks':
    from Pytorch.DeepVO.net.cnn_increase_kernal_size import Net
    from Pytorch.DeepVO.dataset.kitti import KITTIDataSet
elif args.net_architecture == 'cnn-lstm':
    from Pytorch.DeepVO.net.cnn_lstm import Net
    from Pytorch.DeepVO.dataset.kitti_lstm import KITTIDataSet, read_image
else:
    raise Exception('Must give the right cnn architecture')


def run_batch(sample, model, loss_func=None, optimizer=None, phase=None):
    """
    训练、验证：
        run_batch(sample, model, loss_func, optimizer, phase='Train')
        run_batch(sample, model, loss_func, phase='Valid')
        返回估计位姿以及loss
    测试：
        run_batch(sample, model, phase='Test')
        返回估计位姿
    """
    if phase == 'Train':
        model.train()
    else:
        model.eval()  # 启用测试模式，关闭dropout

    img1 = to_var(sample['img1'])  # as for cnn: [bs, 6, H, W], as for cnn-lstm: [N, T, 6, H, W]
    img2 = to_var(sample['img2'])
    label_pre = model(img1, img2)  # [32, 6]
    # conv_out = x_conv.data.cpu().numpy()
    # lstm_out = x_lstm.data.cpu().numpy()
    # print('Conv >>> min: {:.5f}, max: {:.5f}'.format(np.min(conv_out), np.max(conv_out)))
    # print('LSTM >>> min: {:.5f}, max: {:.5f}'.format(np.min(lstm_out), np.max(lstm_out)))

    if phase == 'Train' or phase == 'Valid':
        label = to_var(sample['label'])  # [bs, 6]
        label = label.view(-1, 6)
        loss1 = loss_func(label_pre[:, :3], label[:, :3])
        loss2 = loss_func(label_pre[:, 3:], label[:, 3:])
        loss = loss1 + args.beta * loss2

        # loss_x = loss_func(label_pre[:, 0], label[:, 0])
        # loss_y = loss_func(label_pre[:, 1], label[:, 1])
        # loss_z = loss_func(label_pre[:, 2], label[:, 2])
        # loss_tx = loss_func(label_pre[:, 3], label[:, 3])
        # loss_ty = loss_func(label_pre[:, 4], label[:, 4])
        # loss_tz = loss_func(label_pre[:, 5], label[:, 5])

        if phase == 'Train':
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # bp, compute gradients
            optimizer.step()  # apply gradients

        return loss.data.item(), loss1.data.item(), loss2.data.item(), label_pre.data
        # return loss.data.item(), loss1.data.item(), loss2.data.item(), label_pre.data, \
        #     loss_x.data.item(), loss_y.data.item(), loss_z.data.item(), loss_tx.data.item(), loss_ty.data.item(), loss_tz.data.item()
    else:
        return label_pre.data


def run_batch_2(sample, model, loss_func=None, optimizer=None):
    """
    cnn-lstm 不同time_step一起训练
    """
    model.train()

    loss_mean = []
    loss1_mean = []
    loss2_mean = []
    for sample_batch in sample:
        img1 = to_var(sample_batch['img1'])  # as for cnn: [bs, 6, H, W], as for cnn-lstm: [N, T, 6, H, W]
        img2 = to_var(sample_batch['img2'])
        label_pre = model(img1, img2)  # [32, 6]

        label = to_var(sample_batch['label'])  # [bs, 6]
        label = label.view(-1, 6)
        loss1 = loss_func(label_pre[:, :3], label[:, :3])
        loss2 = loss_func(label_pre[:, 3:], label[:, 3:])
        loss = loss1 + args.beta * loss2

        loss1_mean.append(loss1.data.item())
        loss2_mean.append(loss2.data.item())
        loss_mean.append(loss.data.item())

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # bp, compute gradients
        optimizer.step()  # apply gradients

    loss1_mean = np.mean(loss1_mean)
    loss2_mean = np.mean(loss2_mean)
    loss_mean = np.mean(loss_mean)
    return loss1_mean.data.item(), loss2_mean.data.item(), loss_mean.data.item()


def run_val(model, loss_func, loader):
    """
    验证多个batch，并返回平均误差
    """
    loss_ret = []
    loss1_ret = []
    loss2_ret = []

    for _, sample_v in enumerate(loader):
        loss_v, loss1_v, loss2_v, _ = run_batch(sample=sample_v, model=model, loss_func=loss_func, phase='Valid')
        loss_ret.append(loss_v)
        loss1_ret.append(loss1_v)
        loss2_ret.append(loss2_v)

    loss_mean = np.mean(loss_ret)
    loss1_mean = np.mean(loss1_ret)
    loss2_mean = np.mean(loss2_ret)

    return loss_mean, loss1_mean, loss2_mean


def run_test(model, seq, dir_model=None, epoch=None, dir_time=None):
    """
    训练阶段对一段完整的轨迹进行测试，或者测试阶段直接用于测试

    训练过程中测试：
    1. 计算一段完整场景中所有相对姿态的预测值
    cnn-lstm:
        手动写读图的代码，从而可以处理场景末尾图片序列长度不足一个batch的情况
    cnn:
        采用DataLoader读取，较为方便

    2. 计算绝对姿态，并画出轨迹
    训练阶段保存轨迹图
    测试阶保存轨迹图、相对位姿、绝对位姿
    """
    print('\nTest sequence {:02d} >>>'.format(seq))
    if args.net_architecture == 'cnn-lstm':
        model.eval()
        img_list = glob(dir_data + '/{:02d}/image_2/*.png'.format(seq))
        img_list.sort()
        ip = args.img_pairs
        iter_1 = int(math.floor((len(img_list) - 1) / ip))
        iter_2 = int(math.ceil((len(img_list) - 1) / ip))
        pose_ret = []
        for i in tqdm(np.arange(iter_1)):
            img_seq = []
            for img_path in img_list[i * ip: (i + 1) * ip + 1]:
                img = read_image(img_path)
                img_seq.append(img)
            x1 = np.stack(img_seq[:-1], 0)
            x1 = np.transpose(x1, [0, 3, 1, 2])  # [10, C, H, W]
            x1 = x1[np.newaxis, :, :, :, :]  # [1, 10, C, H, W]
            x1 = to_var(torch.from_numpy(x1))

            x2 = np.stack(img_seq[1:], 0)
            x2 = np.transpose(x2, [0, 3, 1, 2])  # [10, C, H, W]
            x2 = x2[np.newaxis, :, :, :, :]  # [1, 10, C, H, W]
            x2 = to_var(torch.from_numpy(x2))
            pose_out = model(x1, x2)
            pose_ret.extend(pose_out.data.cpu().numpy())

        ns = iter_1 * ip
        if iter_1 != iter_2:
            print('Process for the last {:d} images...'.format(len(img_list) - ns))
            img_seq = []
            for img_path in img_list[ns:]:
                img = read_image(img_path)
                img_seq.append(img)
            x1 = np.stack(img_seq[:-1], 0)
            x1 = np.transpose(x1, [0, 3, 1, 2])  # [10, C, H, W]
            x1 = x1[np.newaxis, :, :, :, :]  # [1, 10, C, H, W]
            x1 = to_var(torch.from_numpy(x1))

            x2 = np.stack(img_seq[1:], 0)
            x2 = np.transpose(x2, [0, 3, 1, 2])  # [10, C, H, W]
            x2 = x2[np.newaxis, :, :, :, :]  # [1, 10, C, H, W]
            x2 = to_var(torch.from_numpy(x2))
            pose_out = model(x1, x2)
            pose_ret.extend(pose_out.data.cpu().numpy())
    else:
        data_set = KITTIDataSet(dir_data=dir_data, dir_label=dir_label, phase='Test', seq=seq)
        loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        pose_ret = []
        for _, sample_batch in enumerate(tqdm(loader)):
            pose_pre = run_batch(sample=sample_batch, model=model, phase='Test')
            pose_ret.extend(pose_pre.cpu().numpy())

    pose_abs = cal_absolute_from_relative(pose_ret)

    if args.phase == 'Test':
        np.savetxt(dir_time + '/pose_{:d}.txt'.format(seq), pose_ret)
        np.savetxt((dir_time + '/{:02d}.txt'.format(seq)), pose_abs)
        plot_from_pose(seq=seq, dir_save=dir_time, pose_abs=pose_abs, args=args)
        print('Save pose and trajectory in {:s}'.format(dir_time))
    else:
        plot_from_pose(seq=seq, dir_save=dir_model, pose_abs=pose_abs, epoch=epoch, args=args)
        print('Save trajectory in {:s}'.format(dir_model))


def main():
    torch.set_default_tensor_type('torch.FloatTensor')
    model = Net()
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu)

    # Set weights
    print('\n========================================')
    print('Phase: {:s}\nNet architecture: {:s}'.format(args.phase, args.net_architecture))
    # if args.net_architecture == 'cnn-lstm':
    #     if args.resume == 'cnn':
    #         print('Restore from CNN: {:s}'.format(dir_restore))
    #         pre_trained_dict = torch.load(dir_restore)
    #         model_dict = model.state_dict()
    #         pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}  # tick the useless dict
    #         model_dict.update(pre_trained_dict)  # update the dict
    #         model.load_state_dict(model_dict)  # load updated dict into the model
    #     elif args.resume == 'lstm' or args.phase == 'Test':
    #         print('Restore from CNN-LSTM: {:s}'.format(dir_restore))
    #         model.load_state_dict(torch.load(dir_restore))
    #     else:
    #         print('Initialize from scratch')
    # else:
    #     if args.resume == 'Yes' or args.phase == 'Test':
    #         print('Restore from CNN: {:s}'.format(dir_restore))
    #         model.load_state_dict(torch.load(dir_restore))
    #     else:
    #         print('Initialize from scratch')
    print('========================================')

    # Start training
    if args.phase == 'Train':
        dir_model, dir_log = pre_create_file_train(model_dir, log_dir, args)
        writer = SummaryWriter(dir_log)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)

        data_set_t = KITTIDataSet(dir_data=dir_data, dir_label=dir_label, samples=args.samples, phase='Train')
        # data_set_v = KITTIDataSet(dir_data=dir_data, dir_label=dir_label, samples=args.samples, phase='Valid')
        loader_t = DataLoader(data_set_t, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        # loader_v = DataLoader(data_set_v, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        step_per_epoch = int(math.floor(len(data_set_t) / loader_t.batch_size))
        step_val = int(math.floor(step_per_epoch / 3))  # 每个epoch验证3次

        for epoch in np.arange(args.epoch_max):
            # adjust_learning_rate(optimizer, epoch, args.lr_base, args.lr_decay_rate, args.epoch_lr_decay)

            # test a complete sequence and plot trajectory
            if epoch != 0 and epoch % args.epoch_test == 0:
                run_test(model, seq=9, dir_model=dir_model, epoch=epoch)
                run_test(model, seq=5, dir_model=dir_model, epoch=epoch)

            loss_list = []  # 记录每个epoch的loss
            loss1_list = []
            loss2_list = []
            for step, sample_t in enumerate(loader_t):
                step_global = epoch * step_per_epoch + step
                tic = time()
                loss, loss1, loss2, _ = run_batch(sample=sample_t, model=model, loss_func=loss_func, optimizer=optimizer, phase='Train')
                hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)
                loss_list.append(loss)
                loss1_list.append(loss1)
                loss2_list.append(loss2)

                # display and add to tensor board
                if (step + 1) % 10 == 0:
                    display_loss_tb(hour_per_epoch, epoch, args, step, step_per_epoch, optimizer, loss, loss1,loss2, loss_list, loss1_list, loss2_list, writer, step_global)

                # if (step + 1) % step_val == 0:
                #     batch_v = int(math.ceil(len(data_set_v) / loader_v.batch_size))
                #     loss_v, loss1_v, loss2_v = run_val(model, loss_func, loader_v)
                #     display_loss_tb_val(batch_v, loss_v, loss1_v, loss2_v, args, writer, step_global)

            # save
            if (epoch + 1) % args.epoch_save == 0:
                print('\nSaving model: {:s}/model-{:d}.pkl'.format(dir_model, epoch + 1))
                torch.save(model.state_dict(), (dir_model + '/model-{:d}.pkl'.format(epoch + 1)))


if __name__ == '__main__':
    main()







