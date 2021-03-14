from glob import glob
import cv2
import os
import sys  
sys.path.append('../')

from src.alg.start_work import start_contours_algorithm


def read_images(path):
    titles = []
    images = []
    count = 0
    format_length = 4
    for image_path in glob(os.path.join(path + 'in/', "*.jpg")):
        titles.append(os.path.basename(image_path)[:-format_length])
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images.append(image_bgr)
        count += 1
    return images, titles


def main():
    date_path = os.getcwd() + '/../' + 'date/'
    # read date
    images, titles = read_images(date_path)
    # start work
    start_contours_algorithm(images, titles, date_path)


if __name__ == "__main__":
    main()