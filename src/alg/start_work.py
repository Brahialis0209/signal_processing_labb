import math
import cv2
import numpy as np

from src.alg.features import seek_chair_contours, seek_w_chair, seek_night_stand_conours
from src.alg.visualization import view_image, draw_all_pictures, draw_message


def start_contours_algorithm(images, titles, date_path):
    yell_hsv_color = [30, 255, 255]
    red_bgr_color = (0, 0, 255)
    title_index = 0
    morphology_kernel_dim = (30, 30)
    contours_curve_dim = 4
    result_path = date_path + 'results/'
    for img in images:
        # translate rgb to hsv
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # limits yellow color in HSV model
        yell_low = np.array([16, 22, 112])
        yell_high = np.array([35, 255, 255])
        # get mask
        curr_mask = cv2.inRange(hsv_img, yell_low, yell_high)
        # make morphology operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morphology_kernel_dim)
        open_mask = cv2.morphologyEx(curr_mask, cv2.MORPH_OPEN, kernel)  # for closing open limits
        hsv_img[open_mask > 0] = yell_hsv_color
        target = cv2.bitwise_and(img, img, mask=open_mask)
        # again take rgb model for algorithm
        RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
        # find contours
        contours, hierarchy = cv2.findContours(open_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_result = img.copy()
        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, red_bgr_color, contours_curve_dim)
        # sort our contours that find biggest area(table, chair, night stand)
        sort_contours = sorted(contours, key=cv2.contourArea)
        cv2.drawContours(img_contours, contours, -1, red_bgr_color, contours_curve_dim)
        # incomment because see all work
        # draw_all_pictures(img, curr_mask, open_mask, hsv_img, gray, img_contours)

        # table have biggest area
        table_contour = sort_contours[-1]
        x, y, w, h = cv2.boundingRect(table_contour)
        x1, y1, w1, h1 = cv2.boundingRect(sort_contours[-2])
        x2, y2, w2, h2 = cv2.boundingRect(sort_contours[-3])
        x3, y3, w3, h3 = cv2.boundingRect(sort_contours[-4])
        rect_dict = {(x1 + w1 / 2, y1 + h1 / 2): sort_contours[-2],
                     (x2 + w2 / 2, y2 + h2 / 2): sort_contours[-3], (x3 + w3 / 2, y3 + h3 / 2): sort_contours[-4]}

        # take contour(chair) which most distant from table
        chair_contour, x_max, y_max = seek_chair_contours(rect_dict, x, y, w, h)
        cv2.drawContours(img_result, chair_contour, -1, red_bgr_color, contours_curve_dim)

        # take other two contour
        night_stand_cnt_left, night_stand_cnt_right, night_stand_left = \
            seek_night_stand_conours(rect_dict, x_max, y_max)
        cv2.drawContours(img_result, night_stand_cnt_left, -1, red_bgr_color, contours_curve_dim)
        cv2.drawContours(img_result, night_stand_cnt_right, -1, red_bgr_color, contours_curve_dim)

        # cv method so that find extreme point in contour(extreme points out night stands)
        rightmost = tuple(night_stand_cnt_left[night_stand_cnt_left[:, :, 0].argmax()][0])
        leftmost = tuple(night_stand_cnt_right[night_stand_cnt_right[:, :, 0].argmin()][0])
        # night stand distant
        night_stand_dist = abs(rightmost[0] - leftmost[0])

        # seek width chair
        w_chair = seek_w_chair(chair_contour)
        # seek extreme point on chair because take into account the scale
        top_chair_point = tuple(chair_contour[chair_contour[:, :, 1].argmin()][0])
        dist_chair_night = top_chair_point[1] - night_stand_left[1]
        bust_koef = dist_chair_night / img.shape[0] + 1
        night_stand_dist *= bust_koef
        print('w_ch: ', w_chair, 'night_stand_dis : ', night_stand_dist)
        if w_chair <= night_stand_dist:
            print("Success")
            draw_message(img_result, "Success")
        else:
            print('Failure')
            draw_message(img_result, "Failure")
        cv2.imwrite(result_path + titles[title_index] + ".png", img_result)
        title_index += 1
        # view_image(img_result, title='result')




