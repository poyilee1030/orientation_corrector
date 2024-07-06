import cv2
import numpy as np
import math
import imutils
import random


def angle_between_points(p1, p2):
    change_x = p2[0] - p1[0]
    change_y = p2[1] - p1[1]
    degree = math.degrees(math.atan2(change_y, change_x))
    return degree


def is_isolate(target_box, all_box_list):
    tx, ty, tw, th = target_box
    offset = int(tw * 0.65)
    observe_x1 = tx - offset
    observe_x2 = tx + tw + offset
    observe_y = ty + th//2
    for box in all_box_list:
        x, y, w, h = box
        left = x
        right = x + w
        top = y
        bottom = y + h
        if left < observe_x1 < right and top < observe_y < bottom:
            return False
        if left < observe_x2 < right and top < observe_y < bottom:
            return False
    return True


class OrientationDetect:

    def __init__(self, parent_path, setting, img_scale, counter):
        self.parent_path = parent_path
        self.setting = setting
        self.img_scale = img_scale
        self.counter = counter

    def auto_canny(self, gray_np_img):
        base = self.setting.orientation_detect_auto_canny_base
        sigma = self.setting.orientation_detect_auto_canny_sigma
        # compute the median of the single channel pixel intensities
        v = np.median(gray_np_img)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (base - sigma) * v))
        upper = int(min(255, (base + sigma) * v))
        # print(f"OrientationDetect, median = {v}, base = {base}, sigma = {sigma}, lower={lower}, upper={upper}")
        edged = cv2.Canny(gray_np_img, lower, upper)
        return edged

    def find_right_box(self, cur_group_list, box_list):
        cur_box = cur_group_list[-1]
        cur_box_x = cur_box[0]
        cur_box_y = cur_box[1]
        cur_box_width = cur_box[2]
        cur_box_height = cur_box[3]
        cur_box_x_center = cur_box_x + cur_box_width // 2
        cur_box_y_center = cur_box_y + cur_box_height // 2

        the_right_box = None
        for box in box_list:
            if cur_box == box:
                continue
            box_x, box_y, box_w, box_h = box
            box_x_center = box_x + box_w // 2
            box_y_center = box_y + box_h // 2
            dis_x = box_x_center - cur_box_x_center
            dis_y = abs(box_y_center - cur_box_y_center)
            avg_h = (cur_box_height + box_h) // 2
            max_w = max(cur_box_width, box_w) * 1.6

            if dis_y < avg_h*0.2 and 0 < dis_x <= max_w:
                the_right_box = box

        if the_right_box is None:
            return cur_group_list
        else:
            cur_group_list.append(the_right_box)
            return self.find_right_box(cur_group_list, box_list)

    def remove_box_from_list(self, new_cur_group_list, remain_box_list):
        for box in new_cur_group_list:
            remain_box_list.remove(box)
        return remain_box_list

    def calculate_score_with_v_lines(self, group, v_lines):
        min_x = group[0][0]
        max_x = group[-1][0] + group[-1][2]

        cut_count = 0
        for line in v_lines:
            cut_x = (line[0] + line[2]) // 2
            if min_x < cut_x < max_x:
                cut_count += 1

        # 都沒被切到=>/1，被切1刀=>/2，依此類推
        score = len(group) / (cut_count + 1)
        return score

    def calculate_candidate_score(self, candidate_list, debug_canvas, v_lines):
        groups_list = []
        remain_box_list = sorted(candidate_list, key=lambda tup: tup[0])

        while remain_box_list:
            current_box = remain_box_list[0]
            cur_group_list = [current_box]
            new_cur_group_list = self.find_right_box(cur_group_list, remain_box_list)
            groups_list.append(new_cur_group_list)
            remain_box_list = self.remove_box_from_list(new_cur_group_list, remain_box_list)

        total_score = 0
        for idx, group in enumerate(groups_list):
            if len(group) == 1:
                if self.setting.debug_mode:
                    box = group[0]
                    x, y, w, h = box
                    sx = x
                    sy = y + h
                    cv2.putText(img=debug_canvas, text='x', org=(sx, sy), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=0.5, color=(0, 0, 255), thickness=1)
            else:
                score = self.calculate_score_with_v_lines(group, v_lines)
                total_score += score
                if self.setting.debug_mode:
                    for j, box in enumerate(group):
                        x, y, w, h = box
                        sx = x
                        sy = y + h
                        cv2.putText(img=debug_canvas, text=str(j), org=(sx, sy), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                    fontScale=0.5, color=(0, 255, 0), thickness=1)

        return total_score, debug_canvas

    def calculate_horizontal_line_count(self, at_img, gray_np_img, v_lines=None):
        if v_lines is None:
            v_lines = []

        img_h, img_w = at_img.shape
        min_area = self.setting.orientation_detect_calculate_count_min_area
        max_area = self.setting.orientation_detect_calculate_count_max_area

        contours = cv2.findContours(at_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(contours) > 19000:
            print(f"len(contours) = {len(contours)}")
            morphology_img = gray_np_img
            kernel = np.ones((3, 3), np.uint8)
            for j in range(1):
                morphology_img = cv2.dilate(morphology_img, kernel, iterations=1)
                morphology_img = cv2.erode(morphology_img, kernel, iterations=1)
            blurred_img = cv2.GaussianBlur(morphology_img, (9, 9), 1.4)
            canny_img = self.auto_canny(blurred_img)
            at_img = canny_img
            img_h, img_w = at_img.shape
            contours = cv2.findContours(at_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            print(f"after blur len(contours) = {len(contours)}")

        if len(contours) > 15000:
            min_area = 289
        elif len(contours) > 11000:
            min_area = 256
        elif len(contours) > 8000:
            min_area = 225

        canvas1 = np.zeros((img_h, img_w, 3), np.uint8)
        size_valid_list = []
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / h
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if 0.5 < aspect < 1.35 and min_area < area < max_area:
                cv2.drawContours(canvas1, [cnt], 0, color, 1)
                cv2.rectangle(canvas1, (x, y), (x+w, y+h), color, 2)
                size_valid_list.append((x, y, w, h))

        if self.setting.debug_mode:
            msg = self.counter.log("OrientationDetect", "calculate_horizontal_line_count", "contour")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", canvas1)

        candidate_list = []
        for idx, box in enumerate(size_valid_list):
            if not is_isolate(box, size_valid_list):
                candidate_list.append(box)

        canvas2 = np.zeros((img_h, img_w, 3), np.uint8)
        if self.setting.debug_mode:
            for idx, box in enumerate(candidate_list):
                x, y, w, h = box
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(canvas2, (x, y), (x + w, y + h), color, 2)

            msg = self.counter.log("OrientationDetect", "calculate_horizontal_line_count", "candidate_contour")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", canvas2)

        score, canvas2 = self.calculate_candidate_score(candidate_list, canvas2, v_lines)

        if self.setting.debug_mode:
            msg = self.counter.log("OrientationDetect", "calculate_horizontal_line_count", "score")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", canvas2)

        return score

    def calculate_orientation(self, gray_np_img):
        # eq_img = cv2.equalizeHist(gray_np_img)
        # if self.setting.debug_mode:
        #     msg = self.counter.log("LineFlow", "calculate_orientation", "equalizeHist")
        #     cv2.imwrite(f"{self.parent_path}/{msg}.jpg", eq_img)

        blurred_img = None
        if self.setting.orientation_detect_enable_blur:
            morphology_img = gray_np_img
            kernel = np.ones((3, 3), np.uint8)
            for j in range(1):
                morphology_img = cv2.dilate(morphology_img, kernel, iterations=1)
                morphology_img = cv2.erode(morphology_img, kernel, iterations=1)
            if self.setting.debug_mode:
                msg = self.counter.log("OrientationDetect", "calculate_orientation", "morphology")
                cv2.imwrite(f"{self.parent_path}/{msg}.jpg", morphology_img)

            blurred_img = cv2.GaussianBlur(morphology_img, (9, 9), 1.4)
            if self.setting.debug_mode:
                msg = self.counter.log("OrientationDetect", "calculate_orientation", "GaussianBlur")
                cv2.imwrite(f"{self.parent_path}/{msg}.jpg", blurred_img)

        if blurred_img is None:
            canny_img = self.auto_canny(gray_np_img)
        else:
            canny_img = self.auto_canny(blurred_img)

        if self.setting.debug_mode:
            msg = self.counter.log("OrientationDetect", "calculate_orientation", "auto_canny")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", canny_img)
        line_count1 = self.calculate_horizontal_line_count(canny_img, gray_np_img, [])

        rotated_gray_np_img = imutils.rotate_bound(gray_np_img, -90)
        rotated_canny = imutils.rotate_bound(canny_img, -90)
        if self.setting.debug_mode:
            msg = self.counter.log("OrientationDetect", "calculate_orientation", "rotated")
            cv2.imwrite(f"{self.parent_path}/{msg}.jpg", rotated_canny)
        line_count2 = self.calculate_horizontal_line_count(rotated_canny, rotated_gray_np_img, [])

        if line_count1 >= line_count2:
            return False
        else:
            return True
