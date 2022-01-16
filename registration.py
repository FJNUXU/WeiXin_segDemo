# -*- coding = utf-8 -*-
# @Time : 2021/7/10 9:44
# @Author : 阮智霖
# @File : registration.py
# @Software: PyCharm


import cv2
import numpy as np


# 7.图像配准结果
def img_match(ori_img, skew_img):
    """
    三、颜色识别
    """
    # 检测器。ORB AKAZE可以，KAZE需要将BFMatcher中汉明距离换成cv2.NORM_L2
    detector = cv2.AKAZE_create()

    # kp1 = detector.detect(orig_image, None)
    # kp2 = detector.detect(skewed_image, None)
    # kp1, des1 = detector.compute(orig_image, kp1)#计算出描述子
    # kp2, des2 = detector.compute(skewed_image, kp2)

    # 也可以一步直接同时得到特征点和描述子
    # find the keypoints and descriptors with ORB

    kp1, des1 = detector.detectAndCompute(ori_img, None)
    kp2, des2 = detector.detectAndCompute(skew_img, None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 暴力匹配,True判断交叉验证
    # matches = bf.match(des1, des2)  # 利用匹配器得到相近程度
    # matches = sorted(matches, key=lambda x: x.distance)  # 按照描述子之间的距离进行排序
    # img3 = cv2.drawMatches(ori_img, kp1, skew_img, kp2, matches[:50], None, (255, 0, 0), flags=2)  # 找到50个匹配对

    # Apply ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # 为了得到最近邻和次近邻，这里交叉验证要设置为false
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    # lenofgood=0
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])  # good.append(m)
            # lenofgood=lenofgood+1

    # matches2 = sorted(good, key=lambda x: x.distance)  # 按照描述子之间的距离进行排序
    # cv2.drawMatchesKnn expects list of lists as matches.
    # img4 = cv2.drawMatches(orig_image, kp1, skewed_image, kp2, matches2[:50],None, flags=2)
    # img4 = cv2.drawMatchesKnn(orig_image, kp1, skewed_image, kp2, good[:50],None, flags=2)
    # img4 = cv2.drawMatchesKnn(ori_img, kp1, skew_img, kp2, good, None, flags=2)
    lenofgood = len(good)  # good长度为24

    des1fromgood = np.ones((lenofgood, 61), dtype=np.uint8)
    # des2fromgood=[]
    des2fromgood = np.ones((lenofgood, 61), dtype=np.uint8)
    for num in range(0, 24):
        for i in good:
            for j in i:
                # good是list组成的list，每个子list中是DMatch
                des1fromgood[num] = des1[j.queryIdx]  # 将good中的特征点对应的描述子形成新的描述子
                # des2fromgood.append(des2[j.trainIdx])
                des2fromgood[num] = des2[j.trainIdx]

    # cv2.imwrite("AKAZETest.jpg", img3)
    # cv2.namedWindow('AKAZETest', cv2.WINDOW_NORMAL)
    # cv2.imshow('AKAZETest', img3)
    # cv2.imwrite("AKAZETestRatio.jpg", img4)
    # cv2.namedWindow('AKAZETestRatio', cv2.WINDOW_NORMAL)
    # cv2.imshow('AKAZETestRatio', img4)

    # H,mask=cv2.findHomography(des1fromgood,des2fromgood,cv2.RANSAC,5.0)#,None,None,None,None,None)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    MIN_MATCH_COUNT = 10
    # print(len(good))
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # ss = M[0, 1]
        # sc = M[0, 0]
        # scaleRecovered = math.sqrt(ss * ss + sc * sc)
        # thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        # logging打印出通过矩阵计算出的尺度缩放因子和旋转因子
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        '''
        logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        # 配置之后将log文件保存到本地
        logging.info("MAP: Calculated scale difference: %.2f, "
                     "Calculated rotation difference: %.2f" %
                     (scaleRecovered, thetaRecovered))
        '''
        # deskew image
        im_out = cv2.warpPerspective(skew_img, np.linalg.inv(M), (ori_img.shape[1], ori_img.shape[0]))
        # result = cv2.warpPerspective(ori_img, M, (skew_img.shape[1], skew_img.shape[0]))
        return im_out


def hist_match(src, dst, mask):
    res = np.zeros_like(dst)
    # cdf 为累计分布
    cdf_src = np.zeros((3, 256))
    cdf_dst = np.zeros((3, 256))
    cdf_res = np.zeros((3, 256))
    kw = dict(bins=256, range=(20, 256), normed=True)
    for ch in range(3):
        his_src, _ = np.histogram(src[:, :, ch], **kw)
        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
        cdf_src[ch] = np.cumsum(his_src)
        cdf_dst[ch] = np.cumsum(hist_dst)
        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side='left')
        # print(index)
        np.clip(index, 0, 255, out=index)
        print(len(index[dst[:, :, ch]]))
        res[:, :, ch] = index[dst[:, :, ch]]
        his_res, _ = np.histogram(res[:, :, ch], **kw)
        cdf_res[ch] = np.cumsum(his_res)
    return res, (cdf_src, cdf_dst, cdf_res)


def mean_std(img1, img2, red2):
    # img1 = cv2.imread("chz_0.jpg")
    # img2 = cv2.imread("chz_3.jpg")
    #
    # red1 = cv2.imread("chz_1.jpg")
    # red2 = cv2.imread("chz_2.jpg")
    # red1_lab = cv2.cvtColor(red1, cv2.COLOR_BGR2LAB)
    red2_lab = cv2.cvtColor(red2, cv2.COLOR_BGR2LAB)
    red2_lab = np.array(red2_lab)  # 把图像转成数组格式img = np.asarray(image)
    # shape = img_array.shape
    #
    # h1, w1, c1 = img1.shape
    # h2, w2, c2 = img2.shape

    # img1 = cv2.resize(img1, (int(h1 * 0.3), int(w1 * 0.3)))
    # img2 = cv2.resize(img2, (int(h2 * 0.3), int(w2 * 0.3)))

    height, width, channel = img1.shape

    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img1_lab = np.array(img1_lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    img2_lab = np.array(img2_lab)


    def non_zero_mean(a):
        mean = np.nanmean(np.where(np.isclose(a, 0), np.nan, a))
        return mean

    def non_zero_std(a):
        std = np.nanstd(np.where(np.isclose(a, 0), np.nan, a))
        return std

    def get_avg_std(image):
        avg = []
        std = []

        img_avg_l = non_zero_mean(image[:, :, 0])
        img_std_l = non_zero_std(image[:, :, 0])

        img_avg_a = non_zero_mean(image[:, :, 1])
        img_std_a = non_zero_std(image[:, :, 1])

        img_avg_b = non_zero_mean(image[:, :, 2])
        img_std_b = non_zero_std(image[:, :, 2])

        avg.append(img_avg_l)
        avg.append(img_avg_a)
        avg.append(img_avg_b)

        std.append(img_std_l)
        std.append(img_std_a)
        std.append(img_std_b)

        return avg, std

    # def cv_show(name, img):
    #     cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    #     cv2.imshow(name, img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    ori_avg, ori_std = get_avg_std(img1_lab)
    # print("治前均值", ori_avg)
    # print("治前方差", ori_std)
    img_avg, img_std = get_avg_std(img2_lab)
    # print("治后均值", img_avg)
    # print("治后方差", img_std)

    # for i in range(0, height):
    #     for j in range(0, width):
    #         for k in range(0, channel):
    #             pos = red2_lab[i, j, k]
    #             pos = (pos - img_avg[k]) * (ori_std[k] / img_std[k]) + ori_avg[k]
    #             pos = 0 if pos < 0 else pos
    #             pos = 255 if pos > 255 else pos
    #             red2_lab[i, j, k] = pos

    for k in range(0, channel):
        # print(red2_lab[:, :, k][np.where(red2_lab[:, :, k] != 0)])
        # print(len(red2_lab[:, :, k][np.where(red2_lab[:, :, k] != 0)]))
        # xy = [np.where(red2_lab[:, :, k] != 0)]
        # print([np.where(red2_lab[:, :, k] != 0)])
        # print([np.where(red2_lab[:, :, k] != 0)])
        # print(img_avg[k])
        red2_lab[:, :, k] = (red2_lab[:, :, k] - img_avg[k]) * (ori_std[k] / img_std[k]) + ori_avg[k]
        # pos = (pos - img_avg[k]) * (ori_std[k] / img_std[k]) + ori_avg[k]
        # pos = 0 if pos < 0 else pos
        # pos = 255 if pos > 255 else pos

    red_tran = cv2.cvtColor(red2_lab, cv2.COLOR_Lab2BGR)

    return red_tran
