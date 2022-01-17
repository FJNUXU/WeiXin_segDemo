import cv2
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def chart(img1, img2, img3):
    list_1, g_min, index = division_origin(img1)
    list_2 = division_after(img2, g_min, index)
    list_3 = division_after(img3, g_min, index)

    per_1 = []
    per_2 = []
    per_3 = []
    for i in list_1:
        percent = i / sum(list_1)
        per_1.append(percent)

    for i in list_2:
        percent = i / sum(list_2)
        per_2.append(percent)

    for i in list_3:
        percent = i / sum(list_3)
        per_3.append(percent)

    ws1 = stats.wasserstein_distance(per_3, per_1)
    ws2 = stats.wasserstein_distance(per_3, per_2)
    print("治疗前红斑与正常皮肤沃氏距离：", ws1)
    print("治疗后红斑与正常皮肤沃氏距离：", ws2)
    # print("本次治疗治愈率为：" + str("%.2f" % (abs((ws1 - ws2) / ws1) * 100 / 2)) + "%")
    effect = str("%.2f" % ((ws1 - ws2) / ws1 * 100 / 2)) + "%"
    ws1 = float("%.6f" % ws1)
    # len_list = [list_1, list_2, list_3]
    len_list = [per_1, per_2, per_3]

    return len_list, ws1, effect


# 治疗前红斑分区间并统计
def division_origin(self):
    img_gray = cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
    img_Lab = cv2.cvtColor(self, cv2.COLOR_BGR2Lab)
    height, width, _ = img_Lab.shape
    L = img_Lab[:, :, 0]
    xy = np.where(L >= 20, 1, 0)
    img_red_gray = xy * img_gray
    g_min = np.nanmin(np.where(np.isclose(img_red_gray, 0), np.nan, img_red_gray))
    g_max = np.max(img_gray)
    # print("g_max:", g_max)
    # print("g_min:", g_min)
    # g_min = np.min(img_red_gray)
    interval = np.round((g_max - g_min) / 10)
    # print("interval:", interval)
    # img_red_gray = img_red_gray.flatten()
    img_red_gray = img_gray[np.where(img_gray >= g_min)]
    index = pd.interval_range(g_min, None, 10, interval)
    list_origin_red = pd.cut(img_red_gray, index)
    list_origin_red = list(list_origin_red.value_counts())
    # for i in range(10):
    #     list_origin_red.append(
    #         [dot for dot in img_red_gray if g_min + i * interval < dot < g_min + (i + 1) * interval])

    return list_origin_red, g_min, index


# 根据治疗前图片的区间数据来统计治疗后及正常皮肤的区间数据
def division_after(self, g_min, index):
    after_gray = cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
    # img_red_gray = after_gray.flatten()
    img_red_gray = after_gray[np.where(after_gray >= g_min)]
    list_after_red = pd.cut(img_red_gray, index)
    list_after_red = list(list_after_red.value_counts())

    return list_after_red


def draw_contour(image, gray):
    ret, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(hierarchy)
    # print(np.array(contours).shape)

    # # 依据平均灰度来筛选绘制轮廓
    # mean_gray = []
    # for c in range(len(contours)):
    #     mask = np.zeros(gray.shape, np.uint8)
    #
    #     mean = cv2.mean(gray, mask)  # 计算每个轮廓灰度平均值
    #     # cv_show("res4", res4)
    #     mean_gray.append(mean[0])
    #
    # used_gray = []  # 存储最大，次大等平均灰度值索引
    # # used_gray.append(mean_gray.index(max(mean_gray)))
    # # Max_gray = np.max(mean_gray)
    # for i in mean_gray:
    #     if abs(i - max(mean_gray)) <= 20:
    #         mean_gray_id = mean_gray.index(i)
    #         used_gray.append(mean_gray_id)
    #
    # print(used_gray)

    mean_boxs = []
    # res5 = image.copy()
    for contour in contours:
        # res5 = cv2.drawContours(res5, contours, -1, (0, 0, 255), 5)
        mean_rect = cv2.minAreaRect(contour)
        # print("mean_rect", mean_rect)

        mean_box = cv2.boxPoints(mean_rect)
        mean_box = np.int0(mean_box)
        mean_boxs.append(mean_box)

    # 画外接矩形
    # draw_fin2 = image.copy()
    # res2 = cv2.drawContours(draw_fin2, mean_boxs, -1, (0, 0, 255), 5)
    # 画外接圆
    res3 = image.copy()
    for box in mean_boxs:
        # 外接圆
        (x, y), radius = cv2.minEnclosingCircle(box)
        center = (int(x), int(y))
        radius = int(radius)
        res3 = cv2.circle(res3, center, radius, (255, 255, 255), -1)

    res3 = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY)
    ret, res4 = cv2.threshold(res3, 254, 255, cv2.THRESH_BINARY)
    fin_mask = cv2.bitwise_and(image, image, mask=res4)

    return res4, fin_mask  # 返回掩膜二值图和掩膜BGR图

# img1 = cv2.imread("img_fin1.jpg")
# img2 = cv2.imread("img_fin2.jpg")
# img3 = cv2.imread("img_fin3.jpg")
# len_list, ws1, effect = chart(img1, img2, img3)
# N = 10
# x = np.arange(10)
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.subplot(131)
# plt.bar(x, len_list[0])
# plt.title("Treat_before")
# plt.subplot(132)
# plt.bar(x, len_list[1])
# plt.title("Treat_after")
# plt.subplot(133)
# plt.bar(x, len_list[2])
# plt.title("Normal")
# plt.show()
# print("ws", ws1)
# print("effect", effect)


# 治疗前红斑分区间并统计
def division_origin_a(self):
    # img_gray = cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
    img_Lab = cv2.cvtColor(self, cv2.COLOR_BGR2Lab)
    # height, width, _ = img_Lab.shape
    # L = img_Lab[:, :, 0]
    # xy = np.where(L >= 20, 1, 0)
    # img_red_gray = xy * img_gray
    a = img_Lab[:, :, 1]
    a_min = 129  # 黑色a值为128,计数需要排除黑色
    a_max = np.max(a)
    # print("g_max:", g_max)
    # print("g_min:", g_min)
    # g_min = np.min(img_red_gray)
    interval = np.round((a_max - a_min) / 10)
    # print("interval:", interval)
    # img_red_gray = img_red_gray.flatten()
    img_red_gray = a[np.where(a >= a_min)]
    index = pd.interval_range(a_min, None, 10, interval)
    list_origin_red = pd.cut(img_red_gray, index)
    list_origin_red = list(list_origin_red.value_counts())
    # for i in range(10):
    #     list_origin_red.append(
    #         [dot for dot in img_red_gray if g_min + i * interval < dot < g_min + (i + 1) * interval])

    return list_origin_red, a_min, index


# 根据治疗前图片的区间数据来统计治疗后及正常皮肤的区间数据
def division_after_a(self, a_min, index):
    after_a = cv2.cvtColor(self, cv2.COLOR_BGR2Lab)
    # img_red_gray = after_gray.flatten()
    img_red_gray = after_a[np.where(after_a >= a_min)]
    list_after_red = pd.cut(img_red_gray, index)
    list_after_red = list(list_after_red.value_counts())

    return list_after_red


def chart_a(red1, red2, normal1, normal2):
    list_1, g_min, index = division_origin_a(red1)
    list_2 = division_after_a(red2, g_min, index)
    list_3 = division_after_a(normal1, g_min, index)
    list_4 = division_after_a(normal2, g_min, index)

    per_1 = []
    per_2 = []
    per_3 = []
    per_4 = []
    for i in list_1:
        percent = i / sum(list_1)
        per_1.append(percent)

    for i in list_2:
        percent = i / sum(list_2)
        per_2.append(percent)

    for i in list_3:
        percent = i / sum(list_3)
        per_3.append(percent)

    for i in list_4:
        percent = i / sum(list_4)
        per_4.append(percent)

    ws1 = stats.wasserstein_distance(per_3, per_1)
    ws2 = stats.wasserstein_distance(per_4, per_2)
    print("治疗前红斑与正常皮肤沃氏距离：", ws1)
    print("治疗后红斑与正常皮肤沃氏距离：", ws2)
    # print("本次治疗治愈率为：" + str("%.2f" % (abs((ws1 - ws2) / ws1) * 100 / 2)) + "%")
    effect = str("%.2f" % ((ws1 - ws2) / ws1 * 100 / 2)) + "%"
    ws1 = float("%.6f" % ws1)
    # len_list = [list_1, list_2, list_3]
    len_list = [per_1, per_2, per_3]

    return len_list, ws1, effect







