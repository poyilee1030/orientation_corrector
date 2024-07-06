import cv2
import numpy as np
import math
import imutils
import utils


class LineFlow:

    def __init__(self, parent_path, setting, img_scale, counter):
        self.parent_path = parent_path
        self.setting = setting
        self.img_scale = img_scale
        self.counter = counter

        self.horizontal_ratio_size = 100
        self.limited_length_scale = 0.14

    def calculate_angle(self, gray_np_img):
        img_h, img_w = gray_np_img.shape
        cx = img_w // 2
        cy = img_h // 2
        half_diagonal = np.sqrt(cx*cx+cy*cy)

        draw_img = gray_np_img.copy()
        if len(draw_img.shape) == 2:
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)

        # gray_image = np_image
        # if len(np_image.shape) == 3:
        #     max_rgb_image = max_rgb_filter(np_image)
        #     gray_image = cv2.cvtColor(max_rgb_image, cv2.COLOR_BGR2GRAY)

        bitwise_img = cv2.bitwise_not(gray_np_img)
        if self.setting.debug_mode:
            msg = self.counter.log("LineFlow", "calculate_angle", "bitwise_not")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", bitwise_img)

        at_img = cv2.adaptiveThreshold(bitwise_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       self.setting.rotation_lineflow_adaptive_threshold_block_size, -2)
        if self.setting.debug_mode:
            msg = self.counter.log("LineFlow", "calculate_angle", "adaptive_threshold")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", at_img)

        kernel_size = img_w // self.horizontal_ratio_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        morphology_img = np.copy(at_img)
        for j in range(0, 5):
            morphology_img = cv2.erode(morphology_img, kernel, iterations=2)
            morphology_img = cv2.dilate(morphology_img, kernel, iterations=2)
        if self.setting.debug_mode:
            msg = self.counter.log("LineFlow", "calculate_angle", "morphology")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", morphology_img)

        contours = cv2.findContours(morphology_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h_boxes = []
        for idx, c in enumerate(contours):
            # (x, y, w, h) = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            _, (w, h), angle = rect
            box = cv2.boxPoints(rect)
            four_point = np.int0(box)

            if abs(angle) > 45:
                box_w = h
                box_h = w
            else:
                box_w = w
                box_h = h

            if box_w > img_w / 20:
                h_boxes.append(four_point)
                draw_img = cv2.drawContours(draw_img, [four_point], -1, (0, 0, 255), 1)

        if self.setting.debug_mode:
            msg = self.counter.log("LineFlow", "calculate_angle", "draw_contours")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", draw_img)

        weight_list = []
        angle_list = []
        for index, four_point in enumerate(h_boxes):
            p1, p2, p3, p4 = utils.order_points(four_point)
            is_line_suitable = True

            line_segment = (p3[0] - p4[0], p3[1] - p4[1])
            two_point_distance = np.linalg.norm(np.array(line_segment))
            length_low_bound = img_w * self.limited_length_scale
            if two_point_distance < length_low_bound:
                is_line_suitable = False

            if is_line_suitable:
                lcx = (p3[0] + p4[0]) // 2
                lcy = (p3[1] + p4[1]) // 2
                radius = np.sqrt((cx - lcx) * (cx - lcx) + (cy - lcy) * (cy - lcy))
                weight = 1 - radius / half_diagonal
                theta = math.degrees(math.atan((p3[1] - p4[1]) / (p3[0] - p4[0])))
                weight_list.append(weight)
                angle_list.append(theta)
                if self.setting.debug_mode:
                    point_start = (int(p3[0]), int(p3[1]))
                    point_end = (int(p4[0]), int(p4[1]))
                    draw_img = cv2.line(draw_img, point_start, point_end, (0, 255, 0), 3)

        if self.setting.debug_mode:
            msg = self.counter.log("LineFlow", "calculate_angle", "draw_suitable_line")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", draw_img)

        rotate_angle = 0
        count_angle_list = len(angle_list)
        if count_angle_list == 0:
            pass
        elif count_angle_list == 1:
            rotate_angle = angle_list[0]
        else:
            # https://www.askpython.com/python/examples/how-to-determine-outliers
            reserve_weight_list = []
            reserve_angle_list = []
            if 1 < count_angle_list <= 6:
                # 使用 z-score 來判斷 outlier
                mean = np.mean(angle_list)
                std = np.std(angle_list)
                threshold = 1.8
                for idx, angle in enumerate(angle_list):
                    z_score = (angle - mean) / std
                    if abs(z_score) < threshold:
                        weight = weight_list[idx]
                        reserve_weight_list.append(weight)
                        reserve_angle_list.append(angle)
            else:
                # count_angle_list > 6:
                # 使用 Inter Quartile Range (IQR) 來判斷 outlier
                reserve_weight_list = []
                reserve_angle_list = []
                q1 = np.percentile(angle_list, 25)
                q3 = np.percentile(angle_list, 75)
                iqr = q3 - q1
                lower_limit = q1 - 1.5 * iqr
                upper_limit = q3 + 1.5 * iqr
                for idx, angle in enumerate(angle_list):
                    if lower_limit < angle < upper_limit:
                        weight = weight_list[idx]
                        reserve_weight_list.append(weight)
                        reserve_angle_list.append(angle)

            # 舊方法，直接平均所有角度
            # rotate_angle = np.average(reserve_angle_list) if reserve_angle_list else 0
            # 新方法，離中心較遠的線段給較低的權重，將權重正規畫後相乘取得角度
            weight_sum = sum(reserve_weight_list)
            normalized_weight_list = [x / weight_sum for x in reserve_weight_list]
            for idx, angle in enumerate(reserve_angle_list):
                weight = normalized_weight_list[idx]
                rotate_angle += weight * angle

        return rotate_angle
