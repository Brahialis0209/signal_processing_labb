import math
import cv2
import numpy as np



def max_dist(x_k, y_k, x, y):
    return math.sqrt((x - x_k) ** 2 + (y - y_k) ** 2)


def seek_chair_contours(rect_dict, x, y, w, h):
    x_max, y_max = 0, 0
    max_len = 0
    for x_k, y_k in rect_dict.keys():
        curr_max = max_dist(x_k, y_k, x + w / 2, y + h / 2)
        if curr_max >= max_len:
            max_len = curr_max
            x_max = x_k
            y_max = y_k
    min_contour = rect_dict[(x_max, y_max)]
    return min_contour, x_max, y_max


def seek_night_stand_conours(rect_dict, x_max, y_max):
    night_stand_1 = (0, 0)
    night_stand_2 = (0, 0)
    for x_k, y_k in rect_dict.keys():
        if (x_k, y_k) != (x_max, y_max) and night_stand_1 == (0, 0):
            night_stand_1 = (x_k, y_k)
        elif (x_k, y_k) != (x_max, y_max) and night_stand_2 == (0, 0):
            night_stand_2 = (x_k, y_k)

    if night_stand_1[0] < night_stand_2[0]:
        night_stand_left = night_stand_1
        night_stand_right = night_stand_2
    else:
        night_stand_left = night_stand_2
        night_stand_right = night_stand_1
    return rect_dict[night_stand_left], rect_dict[night_stand_right], night_stand_left


def seek_w_chair(chair_contour):
    rect = cv2.minAreaRect(chair_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, z, d = box
    x_coords = [x[0], y[0], z[0], d[0]]
    y_coords = [x[1], y[1], z[1], d[1]]
    chair_lengths = []
    for i in range(4):
        for j in range(4):
            if i != j:
                chair_lengths.append(max_dist(x_coords[i], y_coords[i], x_coords[j], y_coords[j]))
    chair_lengths = set(chair_lengths)
    chair_lengths = sorted(list(chair_lengths))
    return chair_lengths[0]