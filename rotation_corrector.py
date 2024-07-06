import os
import cv2
import json
import math
import glob
import imutils
import numpy as np
from counter import Counter
from implement.line_flow import LineFlow
from implement.orientation_detect import OrientationDetect
from setting import Setting
import utils


class RotationCorrector:
    class_name = "RotationCorrector"

    def __init__(self, parent_path, setting, img_scale, counter):
        self.counter = counter
        self.angle_corrector = LineFlow(parent_path, setting, img_scale, counter)
        self.orientation_corrector = OrientationDetect(parent_path, setting, img_scale, counter)
        self.last_calculated_angle = 0

    def detect_orientation_should_fix(self, np_img):
        shape_size = len(np_img.shape)
        if shape_size == 3:
            gray_np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_np_img = np_img
        return self.orientation_corrector.calculate_orientation(gray_np_img)

    def do_correction(self, ori_np_img):
        gray_ori_np_img = cv2.cvtColor(ori_np_img, cv2.COLOR_BGR2GRAY)
        self.last_calculated_angle = self.angle_corrector.calculate_angle(gray_ori_np_img)
        rotate_ori_np_img = self.do_rotate(ori_np_img, self.last_calculated_angle)
        return rotate_ori_np_img

    def do_rotate(self, np_img, angle, crop=False, center_point=None):
        height, width = np_img.shape[:2]

        if center_point is None:
            center = (width / 2, height / 2)
        else:
            center = center_point

        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        if crop:
            bound_w = width
            bound_h = height
        else:
            radians = math.radians(angle)
            sin = math.sin(radians)
            cos = math.cos(radians)
            bound_w = int((height * abs(sin)) + (width * abs(cos)))
            bound_h = int((height * abs(cos)) + (width * abs(sin)))
            rotation_matrix[0, 2] += ((bound_w / 2) - center[0])
            rotation_matrix[1, 2] += ((bound_h / 2) - center[1])

        rotate_np_img = cv2.warpAffine(np_img, rotation_matrix, (bound_w, bound_h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotate_np_img


def unit_test_v2(folder_path):
    if not os.path.exists(folder_path):
        return

    img_list = []
    jpg_list = glob.glob(folder_path + "/*.jpg")
    jpeg_list = glob.glob(folder_path + "/*.jpeg")
    png_list = glob.glob(folder_path + "/*.png")
    img_list.extend(jpg_list)
    img_list.extend(jpeg_list)
    img_list.extend(png_list)
    img_list = sorted(img_list)

    pred_dict = {}
    for image_path in img_list:
        print(f"handling path = {image_path}")
        ori_np_img = cv2.imread(image_path)
        normalized_np_img, _ = utils.limit_image_with_max_length(ori_np_img, 1920)

        counter = Counter()
        setting = Setting(True)
        setting.orientation_detect_calculate_count_min_area = 121
        setting.orientation_detect_calculate_count_max_area = 2500
        parent_dir = os.path.dirname(image_path)
        basename_ext = os.path.basename(image_path)
        basename, ext = os.path.splitext(basename_ext)
        save_dir = parent_dir + "/" + basename
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        rt_object = RotationCorrector(save_dir, setting, 1.0, counter)

        rotated_np_img = rt_object.do_correction(normalized_np_img)
        should_rotate = rt_object.detect_orientation_should_fix(rotated_np_img)
        if should_rotate:
            rotated_np_img = imutils.rotate_bound(rotated_np_img, -90)
        dst_path = parent_dir + "/" + basename + "_result" + ext
        cv2.imwrite(dst_path, rotated_np_img)


if __name__ == '__main__':
    demo_path = "./demo"
    unit_test_v2(demo_path)
