import argparse
import cv2
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import detect_final
import registration
import segmentation
import reco_color_test
import os


def cv_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1.拍摄图像
def img2origin(img):
    origin = cv2.imread(img)
    return origin


# 2.拍摄图像（网格）
def img2Lab(img):
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    return Lab
    # cv2.imwrite("")


# 8.光线校正结果
# 9.治疗效果可视化图表
# 10.红斑区域面积
# 11.沃氏距离
# 12.治疗效果


if __name__ == "__main__":
    today_time = datetime.date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="UI_test")
    parser.add_argument('--input_img', type=str, required=True, help='input image to use')
    parser.add_argument('--input_src', type=str, required=True, help='skewed_image')
    parser.add_argument('--output_folder', type=str, required=True, help='image to pack')
    opt = parser.parse_args()
    # print(opt)
    # image = cv2.imread(opt.input_img)
    orig_image = cv2.imread(opt.input_img)
    skewed_image = cv2.imread(opt.input_src)
    ID_file = os.path.split(opt.input_img)[1]
    patientID = ID_file[0:4]
    doctorID = ID_file[5:9]

    out = registration.img_match(orig_image, skewed_image)  # 调整
    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_registration_" + today_time + ".jpg", result)
    cv2.imwrite("folder/" + patientID + "_" + doctorID + "_" + "_00_01_registration_" + today_time + ".jpg", out)
    # end = datetime.datetime.now()
    # reg_time = end - start

    # cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    # cv2.imshow("out", out)
    # cv2.waitKey(0)
    # save = "folder/" + today_time + ".jpg"
    save = "folder/" + patientID + "_" + doctorID + "_" + "_00_01_registration_" + today_time + ".jpg"
    # cv2.imwrite(save, out)
    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_registration_" + today_time + ".jpg", result)

    # result:图像配准结果

    """
    一、目标检测
    """

    result_BGR0 = detect_final.detect_result(opt.input_img)
    # cv2.namedWindow("BGR0", cv2.WINDOW_NORMAL)
    result_BGR1 = detect_final.detect_result(save)  # 确认图片unit or float类型
    cv2.imwrite("folder/" + patientID + "_" + doctorID + "_" + "_00_01_detect_" + today_time + ".jpg", result_BGR1)
    # cv2.namedWindow("BGR1", cv2.WINDOW_NORMAL)
    # cv2.imshow("BGR1", result_BGR0)
    # cv2.waitKey(0)

    """
    二、图像分割
    """
    # start = datetime.datetime.now()
    m0, res0, dst0 = segmentation.get_A_and_threshold(result_BGR0)  # res0: 治前红斑二值图； dst0:治前红斑图
    # res0, res1 = segmentation.threshold_(dst0, result_BGR0)
    # cv2.namedWindow("BGR0", cv2.WINDOW_NORMAL)
    # cv2.imshow("BGR0", res0)
    # cv2.waitKey(0)
    # m1, res1, dst1 = segmentation.get_A_and_threshold(result_BGR1)  # res1: 治后红斑二值图； dst1:治后红斑图
    area_before = segmentation.calcu_area(res0, 0)
    # area_after = segmentation.calcu_area(res1, 0)
    cv2.imwrite("folder/" + patientID + "_" + doctorID + "_" + "_00_01_cluster_" + today_time + ".jpg", m0)
    cv2.imwrite("folder/" + patientID + "_" + doctorID + "_" + "_00_01_threshold_" + today_time + ".jpg", res0)

    # print("治疗前红斑面积：", area_before)
    # print("治疗前红斑面积：", area_after)
    # cv2.namedWindow("m1", cv2.WINDOW_NORMAL)
    # plt.imshow(m1[0], 'gray')
    # plt.show()
    # end = datetime.datetime.now()
    # seg_time = end - start

    """
    三、颜色识别
    """
    # start = datetime.datetime.now()
    # ori_normal_and_red_image:治前正常和红斑；skew_normal_and_red_image：治后正常和红斑；skew_red_image：治后红斑(套割)
    mask, ori_normal_and_red_image = reco_color_test.draw_contour(result_BGR0, res0)
    skew_normal_and_red_image = cv2.bitwise_and(result_BGR1, result_BGR1, mask=mask)
    cv2.imwrite("folder/" + patientID + "_" + doctorID + "_" + "_00_01_erythema_" + today_time + ".jpg", skew_normal_and_red_image)
    skew_red_image = cv2.bitwise_and(result_BGR1, result_BGR1, mask=res0)
    # cv_show("skew_red_image", skew_red_image)
    # cv_show("ori_normal_and_red_image", ori_normal_and_red_image)
    # cv_show("skew_normal_and_red_image", skew_normal_and_red_image)
    res_inv = cv2.bitwise_not(res0)
    # ori_normal_image：治前正常；skew_normal_image：治后正常
    ori_normal_image = cv2.bitwise_and(ori_normal_and_red_image, ori_normal_and_red_image, mask=res_inv)
    skew_normal_image = cv2.bitwise_and(skew_normal_and_red_image, skew_normal_and_red_image, mask=res_inv)
    # cv_show("ori_normal_image", ori_normal_image)
    # cv_show("skew_normal_image", skew_normal_image)
    # 光线校正
    correct_light = registration.mean_std(ori_normal_image, skew_normal_image, skew_red_image)  # 光线校正
    cv2.imwrite("folder/" + patientID + "_" + doctorID + "_" + "_00_01_correct_" + today_time + ".jpg", correct_light)
    # correct_light = np.float32(correct_light)
    # cv2.namedWindow("correct_light", cv2.WINDOW_NORMAL)
    # cv2.imshow("correct_light", correct_light)
    # cv2.waitKey(0)
    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_correct_" + today_time + ".jpg", correct_light)
    # end = datetime.datetime.now()
    # correct_time = end - start

    # ws1:沃氏距离
    # effect: 治愈率
    # start = datetime.datetime.now()
    len_list, ws1, effect = reco_color_test.chart(ori_normal_and_red_image, skew_normal_and_red_image, ori_normal_image)
    # end = datetime.datetime.now()
    # chart_time = end - start
    # start = datetime.datetime.now()
    N = 10
    x = np.arange(10)
    plt.switch_backend('agg')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.subplot(131)
    plt.bar(x, len_list[0])
    plt.title("Treat_before")
    plt.subplot(132)
    plt.bar(x, len_list[1])
    plt.title("Treat_after")
    plt.subplot(133)
    plt.bar(x, len_list[2])
    plt.title("Normal")
    plt.savefig("folder/" + patientID + "_" + doctorID + "_" + "_00_01_chart_" + today_time + ".jpg")
    # plt.show()
    # chart 治愈律可视化图表
    # plt.show()
    # filename2 = opt.output_folder + "0001_0001_00_01_distance_" + today_time + ".json"
    filename2 = "folder/" + patientID + "_" + doctorID + "_" + "_00_01_distance_" + today_time + ".json"
    with open(filename2, 'w') as file_obj:
        json.dump(ws1, file_obj)
    # filename3 = opt.output_folder + "0001_0001_00_01_effect_" + today_time + ".json"
    filename3 = "folder/" + patientID + "_" + doctorID + "_" + "_00_01_effect_" + today_time + ".json"
    with open(filename3, 'w') as file_obj:
        json.dump(effect, file_obj)

    # ans1 = img2origin(orig_image)
    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_000_" + today_time + ".jpg", ans1)
    # cv2.imwrite("folder/" + "0001_0001_00_01_000_" + today_time + ".jpg", ans1)
    # ans2 = img2Lab(orig_image)   # 带网格图片  图片带网格？拍摄时带网格,如何获得不带网格的图片
    # cv2.imwrite(opt.output_folder + "0001_0001_00_00_000_" + today_time + ".jpg", ans2)

    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_detect_" + today_time + ".jpg", result_BGR)

    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_cluster_" + today_time + ".jpg", A_com)

    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_threshold_" + today_time + ".jpg", segment)

    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_erythema_" + today_time + ".jpg", final)

    # cv2.imwrite(opt.output_folder + "0001_0001_00_01_registration_" + today_time + ".jpg", result)

    # filename1 = opt.output_folder + "0001_0001_00_01_area_" + today_time + ".json"
    filename1 = "folder/" + patientID + "_" + doctorID + "_" + "_00_01_area_" + today_time + ".json"
    with open(filename1, 'w') as file_obj:
        json.dump(area_before, file_obj)
