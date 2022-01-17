# -*- coding = utf-8 -*-
# @Time : 2021/7/10 9:52
# @Author : 阮智霖
# @File : segmentation.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt


def OTSU(image, Sigma, Th):
    """
    大津法【除去图片0值】
    :param image
    :param Sigma: -1
    :param Th: 0
    :return: Th
    """
    image_nonzero = image[image != 0]

    for th in range(int(np.min(image_nonzero)), 256):
        bg = image_nonzero[image_nonzero <= th]
        obj = image_nonzero[image_nonzero > th]

        p0 = bg.size / (bg.size + obj.size)
        p1 = obj.size / (bg.size + obj.size)

        m0 = 0 if bg.size == 0 else bg.mean()
        m1 = 0 if obj.size == 0 else obj.mean()

        sigma = p0 * p1 * (m0 - m1) ** 2

        if sigma > Sigma:
            Sigma = sigma
            Th = th

    print("sigma", Sigma)
    print("Th", Th)

    return Th


def get_A_and_threshold(img_BGR):
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    # 3、A分量图
    src = img_LAB

    ##缩小
    scale = 0.1  # 缩小的比例
    height, width = src.shape[:2]
    size = (int(width * scale), int(height * scale))  # bgr
    src = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    ##

    L, A, B = cv2.split(src)  # 分割后单独显示
    # cv2.imshow('L', L)
    # cv2.imshow('A', A)
    # cv2.imshow('B', B)
    # cv2.imwrite('org1_A.jpg', A)
    # print('s.shape:', A.shape)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 4、A分量聚类图
    # 读取原始图像灰度颜色
    img = A

    # 获取图像高度、宽度
    rows, cols = img.shape[:]

    # 图像二维像素转换为一维
    data = img.reshape(rows * cols, 1)
    data = np.float32(data)

    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # flags = cv2.KMEANS_PP_CENTERS

    # K-Means聚类 聚集成...类
    # labels是分类标签，若给定3类，则为0 1 2
    K = 3  # 类别
    compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)

    # 分别提取各类图像，并求均值
    labels_matrix = labels.reshape((img.shape[0], img.shape[1]))  # shape[0]输出行数；shape[1]输出列数

    img_category = np.zeros((K, rows, cols))  # K页矩阵，每页为一类像素
    # print(img_category)
    mean_img_category = np.zeros((K, 1))  # 每类的均值
    min_img_category = np.zeros((K, 1))  # 每类的最小值
    flagMatrix_category = np.zeros((K, rows, cols))  # K页矩阵，每页为一类flag

    for i_labels in range(K):
        flag_category = np.where(labels_matrix == i_labels, 1, 0)  # 第i_labels类标签位置标记为1，否则为0
        img_category_i = flag_category * img  # 两个矩阵逐个元素对应相乘
        img_category[i_labels, :, :] = img_category_i  # 保存第i_labels类标签的图像
        flagMatrix_category[i_labels, :, :] = flag_category

        non_zeros = img_category_i[img_category_i != 0]  # 提取第i_labels类图像中非零元素
        mean_img_category[i_labels] = np.mean(non_zeros)  # 取均值，因为红斑区域值偏大
        min_img_category[i_labels] = np.min(non_zeros)  # 取最小值

        # plt.figure(str(i_labels))
        # plt.imshow(img_category_i, cmap='gray')

    idx_maxMean_imgCatg = np.argmax(mean_img_category)  # 找出具有最大均值的图像类的标签

    # print(mean_img_category)
    # print(idx_maxMean_imgCatg)

    # 绘制均值最大的图像类三维表面图
    img_category_maxMean = img_category[idx_maxMean_imgCatg, :, :]  # 均值最大的图像类的图像
    cluster = np.uint8(img_category_maxMean)
    # img_category_maxMean[img_category_maxMean == 0] = min_img_category[idx_maxMean_imgCatg]
    """
    自适应分割
    """
    # plt.hist(img_category_maxMean[img_category_maxMean != 0].flatten(), np.arange(-0.5, 256, 1), color='g')
    # plt.show()
    # Th = OTSU(img_category_maxMean, -1, 0)

    ret, res = cv2.threshold(img_category_maxMean, 148, 255, cv2.THRESH_BINARY)

    # ret, res = cv2.threshold(img_category_maxMean, 148, 255, cv2.THRESH_BINARY)
    res = cv2.resize(res, (width, height), interpolation=cv2.INTER_AREA)
    # print("res.shape: ", res.shape)
    # print("res.dtype: ", res.dtype)

    res = np.uint8(res)
    # print("img.shape: ", img.shape)
    # print("img.dtype: ", img.dtype)
    # print("res.dtype: ", res.dtype)
    fin = cv2.bitwise_and(img_BGR, img_BGR, mask=res)
    # cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    # cv2.imshow("res", fin)
    # cv2.waitKey(0)
    return cluster, res, fin


def calcu_area(img, b):
    # img = cv2.imread(img)
    xy = np.where(img != b)
    xy = list(xy)[0]
    num = len(xy)
    return num

# 4.dst 聚类结果
# 5.th3 全阈值分割结果
# 6.result 红斑结果


# if __name__ == "__main__":
#      img1 = cv2.imread("folder/result_BGR0.jpg")
#      dst = get_A_and_threshold(img1)
#      cv2.imshow("d1", d1)
#      cv2.imshow("t1", t1)
#      cv2.imshow("r1", r1)
# img = cv2.imread("folder/result_BGR0.jpg")
# c, res, fin = get_A_and_threshold(img)
# cv2.namedWindow("res", cv2.WINDOW_NORMAL)
# cv2.imshow("res", res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()