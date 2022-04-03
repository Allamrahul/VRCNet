import torch.optim as optim
import torch
import numpy as np
import random
import math


def centroidnp(tns, sp=None): # input : 24 x 2048 x 3
    """
    computes centroid of point cloud
    :param tns : bs x pc_size x 3
    :return res : bs x 3
    """
    # print("ip shape", tns.shape)
    bs = tns.shape[0]
    tns = tns.cuda()
    res = torch.FloatTensor(bs, 3).cuda()

    shape = sp if sp else tns.shape[1]

    for x in range(bs):
        a = torch.sum(tns[x][:, 0])/shape
        b = torch.sum(tns[x][:, 1])/shape
        c = torch.sum(tns[x][:, 2])/shape

        res[x] = torch.tensor((a, b, c))
    return res



def tranformation_mtx(p1, p2):  # p2 , p1 is the old centroid
    """
    computes the transformation matrix between source point cloud centroid and destiantion point cloud centroid p2
    :param p1: bs x 3
    :param p2: bs x 3
    :return p: bs x 4 x 4
    """
    bs = p1.shape[0]
    res = torch.FloatTensor(bs, 4, 4).cuda()

    rand_angle = np.random.randint(5, 60)
    radians = math.pi * rand_angle / 180.0

    for x in range(bs):
        res[x] = torch.FloatTensor(
            [[np.cos(radians), -np.sin(radians), 0, p2[x][0] - p1[x][0]],
             [np.sin(radians), np.cos(radians), 0, p2[x][1] - p1[x][1]],
             [0, 0, 1, p2[x][2] - p1[x][2]],
             [0, 0, 0, 1]]
        )

    return res

def final_t(tx, input_cropped1, real_point):
    """
    Accepts tx, the transformation mtx, the input_cropped points and the real_points (ground truth)
    :param : tx : bs x 4 x 4
    :input_cropped1 : (bs, partial_num, 3)
    :real_point : (bs, crop_num, 3)
    """
    input_cropped1 = input_cropped1.cuda()
    real_point = real_point.cuda()

    bs = input_cropped1.shape[0]

    ip_shape = input_cropped1.shape[1]
    gt_shape = real_point.shape[1]

    # creating templates
    input_cropped1_p = torch.FloatTensor(bs, ip_shape, 4).cuda()
    real_point_p = torch.FloatTensor(bs, gt_shape, 4).cuda()

    input_cropped1_f = torch.FloatTensor(bs, ip_shape, 3).cuda()
    real_point_f = torch.FloatTensor(bs, gt_shape, 3).cuda()


    for i in range(bs):
        # padding
        temp = torch.ones((ip_shape, 1)).cuda()
        input_cropped1_p[i] = torch.cat((input_cropped1[i], temp), 1)

        temp = torch.ones((gt_shape, 1)).cuda()
        real_point_p[i] = torch.cat((real_point[i], temp), 1)

        # applying the matrix transformation for both point clouds, selcting the first 3 rows and taking transpose
        input_cropped1_f[i] = torch.matmul(tx[i], input_cropped1_p[i].t())[:3, :].t()
        real_point_f[i] = torch.matmul(tx[i], real_point_p[i].t())[:3, :].t()

    return input_cropped1_f, real_point_f



