import cv2
import datetime
import numpy as np


def print_with_time(debug_string):
    datetime_today_object = datetime.datetime.today()
    today_string_type = datetime_today_object.strftime('%Y.%m.%d_%H:%M:%S')
    print(f"{today_string_type}::processor:{debug_string}")


def order_points(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype="float32")


def resize_image_with_height(np_img, target_height):
    height, width = np_img.shape[:2]
    img_scale = target_height / height
    new_width = int(img_scale * width)
    new_height = target_height
    resized_img = cv2.resize(np_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_img


def limit_image_with_max_length(np_img, max_length):
    if max_length == -1:
        return np_img, 1

    height, width = np_img.shape[:2]
    # cur_max_length = max(width, height)
    # if cur_max_length < max_length:
    #     return np_img, 1

    if width >= height:
        new_width = max_length
        img_scale = new_width / width
        new_height = int(height * img_scale)
    else:
        new_height = max_length
        img_scale = new_height / height
        new_width = int(width * img_scale)

    resized_img = cv2.resize(np_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_img, img_scale


def is_printable_image(np_img):
    small_np_img, scale = limit_image_with_max_length(np_img, 640)
    gray_small_np_img = cv2.cvtColor(small_np_img, cv2.COLOR_BGR2GRAY)

    mask_img = np.zeros_like(gray_small_np_img)
    is_printable = True
    color = ('b', 'g', 'r')
    for i, c in enumerate(color):
        cur_img = small_np_img[:, :, i]
        mask_img[cur_img != gray_small_np_img] = 255
        diff_indices = np.nonzero(mask_img)
        if len(diff_indices[0]) > 0:
            is_printable = False
            break

    return is_printable
