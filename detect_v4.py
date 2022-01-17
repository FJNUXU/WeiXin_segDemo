# -*- coding = utf-8 -*-
# @Time : 2021/8/15 11:30
# @Author : 阮智霖
# @File : detect_v4.py
# @Software: PyCharm


import face_recognition
import cv2
import numpy as np


def cv_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#功能：得除去背景及部分毛发的皮肤
def check_face_yCrCb(image):
    img_BGR = image
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    img_yCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    Cr = img_yCrCb[:, :, 1]  #1.提取Cr分量
    Cr1 = cv2.GaussianBlur(Cr, (5, 5), 0)  # 2.高斯模糊
    ret,  SkinSegment = cv2.threshold(Cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #3.自适应阈值剔除背景
    img1_bg = cv2.bitwise_and(img_RGB, img_RGB, mask=SkinSegment) #4.得到除去背景的掩膜
    img1_bg = cv2.cvtColor(img1_bg, cv2.COLOR_RGB2BGR)
    B = img1_bg[:, :, 2]
    ret1, mask = cv2.threshold(B, 80, 250, cv2.THRESH_TOZERO)  # 5.除去部分毛发
    img1_bg = cv2.bitwise_and(img_RGB, img_RGB, mask=mask)  #6.得除去背景及部分毛发掩膜
    img_face = cv2.cvtColor(img1_bg, cv2.COLOR_BGR2RGB)
    return img_face  #得除去背景及部分毛发的皮肤


#功能：对于侧脸。通过垂直投影精确定位图像中嘴唇的最侧一端x坐标
def getVProjection(image):
    ret, thresh1 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    (h, w) = thresh1.shape  # 返回高和宽
    # a = [0 for z in range(0, w)]  # a = [0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数
    xy1 = np.where(thresh1 == 0, 255, 1)
    xy2 = np.where(thresh1 == 0, 1, 0)
    thresh1 = thresh1 * xy1
    a = np.sum(xy2, axis=0)
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i, j] = 0  # 涂黑
            if (a[j]>a[j+1]):
                return int(j)  #得嘴唇的最侧一端x坐标
    return w-1


#功能：对于侧脸，在去除背景以及毛发的img条件下，进一步去除嘴唇，得侧脸正常皮肤与红斑部分
def side_mouth(img, intx, inty, intw, inth):  # (x,y):左上角坐标；w,h:宽度，高度
    inty = int(inty + inth * 0.6)
    inth = int(inth * 0.4)
    intw = int(intw*0.4)
    img_mouth_BGR = img[inty:(inty + inth), intx:intx + intw]  # 根据先验经验缩小嘴唇所在位置
    img_Gray = cv2.cvtColor(img_mouth_BGR, cv2.COLOR_BGR2GRAY)  # 将img图像转换为灰度图，输出为GrayImage
    mouthXpos1 = getVProjection(img_Gray)  # 调用getVProjection函数进行垂直投影，得嘴唇的最侧一端x坐标，进一步精确嘴唇位置
    img_small_mouth_BGR = img[inty:(inty + inth), intx + mouthXpos1:intx + intw] # 分割出嘴唇
    img_LAB = cv2.cvtColor(img_small_mouth_BGR, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_LAB)
    ret1, mouth_mask = cv2.threshold(A, 145, 250, cv2.THRESH_BINARY_INV)  # 在A分量下凸显嘴唇
    img_small_mouth_BGR = cv2.bitwise_and(img_small_mouth_BGR, img_small_mouth_BGR, mask=mouth_mask)  # 得嘴唇掩膜
    mouth_shape = img_small_mouth_BGR.shape
    mouth_h = int(mouth_shape[0])
    mouth_w = mouth_shape[1]
    img[inty:inty + mouth_h, intx+mouthXpos1:intx+mouthXpos1 + mouth_w] = img_small_mouth_BGR #在去除背景以及部分毛发的img中进一步去除嘴唇
    return img  # 得侧脸正常皮肤与红斑部分


# 功能：对于检测到特征点的图像，将特征点按五官连线
def get_outline(img, ratio, face_landmarks_list):
    for face_landmarks in face_landmarks_list:
        num = 0

        for name, list_of_points in face_landmarks.items():
            last_point_list = []
            first_point_list = []
            point_list = []
            num += 1
            length = len(list_of_points)
            # print(length)
            # print("The {} in this face has the following points: {}".format(name, list_of_points))
            for last_point in list_of_points[0:1]:
                last_point = list(last_point)  # 因为之前将图片放缩，现在将特征点坐标转成列表进行放大（算法得到的是元组）
                # last_point[:] = [int(x * ratio) for x in last_point]
                last_point = tuple(last_point) # 转换回元组，方便五官轮廓的连接
                first_point = last_point
                last_point_list.append(last_point)
                first_point_list.append(first_point)

            for point in list_of_points[1:17]:
                point = list(point)
                # point[:] = [int(x * ratio) for x in point]  # point[:] = [x  for x in point]#
                point_list.append(point)

                if num != 1 and num != 4 and num != 5:
                    point = tuple(point)
                    cv2.line(img, last_point_list[0],
                             point, (255, 255, 255), 10)
                    last_point_list[0] = point
                if num == 1:
                    point = tuple(point)
                    cv2.line(img, last_point_list[0],
                             point, (255, 255, 255), 10)
                    last_point_list[0] = point
            if num != 1 and num != 4 and num != 5:
                point = tuple(point_list[-1])
                cv2.line(img, first_point_list[0],
                         point, (255, 255, 255), 10)
    # cv_show("feature_test", img)

    return img
'''
                if (num != 1 and num != 4 and num != 5):
                    point = tuple(point)
                    cv2.line(img, last_point_list[0],
                             point, (255, 255, 255), 10)
                    last_point_list[0] = point
                if (num == 1):
                    point = tuple(point)
                    cv2.line(img, last_point_list[0],
                             point, (255, 255, 255), 10)
                    last_point_list[0] = point
            if (num != 1 and num != 4 and num != 5):
                point = tuple(point)
                cv2.line(img, first_point_list[0],
                         point, (255, 255, 255), 10)
'''


# 功能：对于检测到特征点的图像，在得到五官轮廓的情况下，利用floodfill算法得掩膜，返回去除五官的人脸
def fill_outline(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 求二值图像
    retv, thresh = cv2.threshold(img_gray, 254, 255, 1)
    im_floodfill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (10, 10), 0)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_floodfill_inv = im_floodfill_inv[0:h, 0:w]
    image = cv2.bitwise_and(image, image, mask=im_floodfill_inv)
    return image

#加载模型并放缩图片
def resize_and_recongnition(img):
    height, width = img.shape[:2]  # 放缩
    ratio = height / 150
    # print("缩放比例:{}".format(ratio))
    img = cv2.resize(img, (int(width / ratio), int(height / ratio)), interpolation=cv2.INTER_CUBIC)
    face_locations = face_recognition.face_locations(img, 1, 'cnn')
    face_landmarks_list = face_recognition.face_landmarks(img)
    return face_locations, face_landmarks_list, ratio


def detect_result(image, ratio, face_location, face_landmarks):
    # 检测到特征点
    # print("检测到特征点：", len(face_landmarks))
    if len(face_landmarks) != 0:
        # img = get_outline(img, face_locations, face_landmarks_list)  #对于检测到特征点的图像，将特征点按五官连线
        img = get_outline(image, ratio, face_landmarks)
        # cv_show("fea_img", img)
        img = fill_outline(img)  # 得去除五官的照片
        # cv_show("img", img)
        result_BGR = check_face_yCrCb(img)  # 再去除背景，得正常皮肤与红斑部分

    elif len(face_location) != 0:
        print("face_location[0]:", face_location)
        top, right, bottom, left = face_location[0]
        top = int(top * ratio)
        right = int(right * ratio)
        bottom = int(bottom * ratio)
        left = int(left * ratio)

        img = check_face_yCrCb(image)  # 去除背景
        intx, inty, intw, inth = left, top, right - left, bottom - top
        result_BGR = side_mouth(img, intx, inty, intw, inth)

    else:
        height, width = image.shape[:2]
        re_h = int(height * 0.2)
        plus_h = int(height * 0.6)
        re_w = 0
        plus_w = int(width * 0.5)
        top, right, bottom, left = (re_h, re_w + plus_w, re_h + plus_h, re_w)
        img = check_face_yCrCb(image)  # 去除背景
        # cv_show("img", img)
        # img = cv2.flip(img, 1)
        intx, inty, intw, inth = left, top, right - left, bottom - top
        result_BGR = side_mouth(img, intx, inty, intw, inth)
        cv_show("res", result_BGR)

    return result_BGR  # 得正常皮肤与红斑部分

'''
    else:
        top, right, bottom, left = (600, 2000, 2300, 560)
        img = check_face_yCrCb(image)  # 去除背景
        cv_show("img", img)

        intx, inty, intw, inth = left, top, right - left, bottom - top
        result_BGR = side_mouth(img, intx, inty, intw, inth)
        cv_show("res", result_BGR)
'''


'''
    else:  # 检测不到特征点
        print("face_location[0]:", face_location)
        top, right, bottom, left = face_location[0]
        top = int(top * ratio)
        right = int(right * ratio)
        bottom = int(bottom * ratio)
        left = int(left * ratio)

        img = check_face_yCrCb(image) #去除背景
        intx, inty, intw, inth = left, top, right - left, bottom - top
        result_BGR = side_mouth(img, intx, inty, intw, inth)  # 再去除侧脸嘴唇，得正常皮肤与红斑部分
    '''




