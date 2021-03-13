import cv2


def view_image(image, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_all_pictures(img, curr_mask, open_mask, hsv_img, gray, img_contours):
    view_image(img, title='original')
    view_image(curr_mask, title='mask')
    view_image(open_mask, title='morphology - open mask')
    view_image(hsv_img, title='with mask')
    view_image(gray, title='to gray')
    view_image(img_contours, title='contours after del little area')


def visul_max_area_cnts(img, sort_contours, red_bgr_color, contours_curve_dim):
    img_contours1 = img.copy()
    img_contours2 = img.copy()
    img_contours3 = img.copy()
    img_contours4 = img.copy()
    cv2.drawContours(img_contours1, sort_contours[-1], -1, red_bgr_color, contours_curve_dim)
    view_image(img_contours1, title='contours after del little area1')
    cv2.drawContours(img_contours2, sort_contours[-2], -1, red_bgr_color, contours_curve_dim)
    view_image(img_contours2, title='contours after del little area2gg')
    cv2.drawContours(img_contours3, sort_contours[-3], -1, red_bgr_color, contours_curve_dim)
    view_image(img_contours3, title='contours after del little area3gg')
    cv2.drawContours(img_contours4, sort_contours[-4], -1, red_bgr_color, contours_curve_dim)
    view_image(img_contours4, title='contours after del little area4gg')


def visul_limits_on_night_stands(img, leftmost, rightmost, box):
    img_buf1 = img.copy()
    img_buf2 = img.copy()
    img_buf1[leftmost[1] - 50: leftmost[1] + 50, leftmost[0] - 50: leftmost[0] + 50] = [0, 255, 0]
    img_buf1[rightmost[1] - 50: rightmost[1] + 50, rightmost[0] - 50: rightmost[0] + 50] = [255, 0, 0]
    cv2.drawContours(img_buf2, [box], 0, (0, 0, 255), 2)
    view_image(img_buf1, title='stand_limits')
    view_image(img_buf2, title='box')

def draw_message(img, message):
    red_bgr_color = (0, 0, 255)
    green_bgr_color = (0, 255, 0)
    if message == "Failure":
        color = red_bgr_color
    else:
        color = green_bgr_color
    cv2.putText(img, message,
                org=(400, 400), color=color,
                fontScale=8, fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness=10)
