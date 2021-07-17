#!/usr/bin/env python3.7
# -*- coding:utf-8 -*-

# @file: analysis_csidata
# @author: Created by HuangZ
# @ide: PyCharm
# @time: 2021/7/5 11:09

import math
import sys
import json
import argparse
import pandas as pd
import numpy as np
import scipy.signal as signal
from sklearn import decomposition, preprocessing  # 数据预处理
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import cv2
import matplotlib.pyplot as plt


def hampel(X):
    """ hampel filter to remove outliers

    :param :one of the wifi packet with all real and imag of subcarriers
    :return:wifi packets after removing outliers

    """
    length = X.shape[0] - 1
    k = 3
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）

    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf


# only plot useful subcarrier channel
select_list = []
select_list += [i for i in range(5, 31)]
select_list += [i for i in range(33, 58)]
select_list += [i for i in range(66, 122)]
select_list += [i for i in range(123, 191)]
select_list.remove(128)


def amplitude_cal(filepath):
    """ calculate amplitude through frequency response

    :param filepath: csv path
    :return:  CSI_amplitude

    """
    df = pd.read_csv(filepath)

    df_rssi = df.loc[:, ['rssi']]
    df_rssi.plot(y=['rssi'])  # plot rssi curve
    plt.axis([0, len(df_rssi.index), -100, 0])

    df_csi = df.loc[:, ['len', 'data']]
    size_x = len(df_csi.index)
    size_y = df_csi.iloc[0]['len'] // 2
    # print('size_x = ', size_x)
    # print('size_y = ', size_y)
    array_csi = np.zeros([size_x, size_y], dtype=np.complex64)
    for x, csi in enumerate(df_csi.iloc):
        csi_raw_data = json.loads(csi['data'])
        for y in range(0, len(csi_raw_data), 2):
            array_csi[x][y //
                         2] = complex(csi_raw_data[y], csi_raw_data[y + 1])  # IQ channel frequency response
            # if csi_raw_data[y] == 0:
            #     array_csi[x][y // 2] = math.degrees(math.pi / 2)
            # else:
            #     array_csi[x][y // 2] =math.degrees(math.atan(csi_raw_data[y + 1] / csi_raw_data[y]))  # csi angle

    array_csi_modulus = abs(array_csi)  # amplitude calculating

    for i in range(len(array_csi_modulus)):
        array_csi_modulus[i] = hampel(array_csi_modulus[i])  # hampel filter
    array_csi_modulus = signal.wiener(array_csi_modulus, (5, 5))  # wiener filter de-noising

    columns = [f"subcarrier{i}" for i in range(0, size_y)]
    df_csi_modulus = pd.DataFrame(array_csi_modulus, columns=columns)

    df_csi_modulus.plot(y=[f"subcarrier{i}" for i in select_list])  # amplitude curve for selected subcarriers
    plt.xlabel('WiFi CSI-Data/Packets')
    plt.ylabel('Amplitude/dBm')
    plt.show()

    return array_csi_modulus


def normalize(csi_array):
    """ normalize all amplitudes into [0, 1]

    :param csi_array:
    :return: normalized csi amplitude array

    """
    array_csi_modulus = preprocessing.scale(csi_array)
    standardScalar = preprocessing.MinMaxScaler()
    standardScalar.fit(array_csi_modulus)
    array_csi_normalized = standardScalar.transform(array_csi_modulus)

    return array_csi_normalized


def decomp(array_csi_modulus):
    """decomposition

    :param array_csi_modulus: normalized csi amplitude array
    :return:
    """
    pca = decomposition.FastICA(n_components=3)  # FA, FICA, MBDL
    CSI = pca.fit_transform(array_csi_modulus)
    CSI = normalize(CSI)
    return CSI


def visualize(array_csi_modulus, color):
    """ visualize amplitudes in 3_Dimension

    :param array_csi_modulus: normalized csi amplitude array
    :param color: the color of scatters
    :return: a 3_D figure on matplotlib
    """

    CSI = decomp(array_csi_modulus)

    x = CSI[:, 0]
    y = CSI[:, 1]
    z = CSI[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, color=color)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    plt.show()


def image_render(array_csi_modulus, length):
    """To render the csi amplitude as an image, you need to select the required subcarriers yourself

    :param array_csi_modulus: normalized csi amplitude array
    :param length:
    :return: image output
    """

    CSI = array_csi_modulus[:, 70:100]  # subcarrier select
    # print(CSI.shape)
    amplitude = np.zeros([length, length])
    for i in range(len(CSI) // length):
        amplitude = CSI[i * length:length + i * length, :]
        amplitude = cv2.normalize(amplitude, None, 0, 255, cv2.NORM_MINMAX)
        plt.figure(figsize=(length, length), dpi=1)
        plt.jet()
        plt.imshow(amplitude)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig('image/amplitude_%d.jpg' % i, dpi=1, bbox_inches='tight')
        print('image %d finished!' % i)


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()

    csi_array0 = amplitude_cal('localization/log.console_test.empty.csv')
    csi_array1 = amplitude_cal('localization/log.console_test.place1.csv')
    csi_array2 = amplitude_cal('localization/log.console_test.place2.csv')
    array_csi_modulus0 = normalize(csi_array0)
    array_csi_modulus1 = normalize(csi_array1)
    array_csi_modulus2 = normalize(csi_array2)
    CSI0 = decomp(array_csi_modulus0)
    CSI1 = decomp(array_csi_modulus1)
    CSI2 = decomp(array_csi_modulus2)

    fig = plt.figure()
    ax = Axes3D(fig)

    x = CSI0[:, 0]
    y = CSI0[:, 1]
    z = CSI0[:, 2]
    ax.scatter(x, y, z, color='r')

    x = CSI1[:, 0]
    y = CSI1[:, 1]
    z = CSI1[:, 2]
    ax.scatter(x, y, z, color='g')

    x = CSI2[:, 0]
    y = CSI2[:, 1]
    z = CSI2[:, 2]
    ax.scatter(x, y, z, color='b')

    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    # visualize(array_csi_modulus)
    # image_render(array_csi_modulus, 30)
