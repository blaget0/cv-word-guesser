import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def crop_image(path):
    src = cv.imread(path)

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #src_gray = cv.blur(src_gray, (3,3))

    threshold = 100
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    #boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        #boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    color = (0, 0, 255)
    for i in range(len(contours)):
        cv.drawContours(drawing, contours_poly, i, color)
    #cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    index = np.argmax(radius)

    #cv.circle(drawing, (int(centers[index][0]), int(centers[index][1])), int(radius[index]), color, 2)

    padding = ceil((2 * radius[index])/28)

    p1 = (int(centers[index][0] + radius[index] + padding), int(centers[index][1] + radius[index] + padding))
    p2 = (int(centers[index][0] - radius[index] - padding), int(centers[index][1] - radius[index] - padding))

    cv.rectangle(drawing, p1, p2, color, 2)

    cropped_image = src_gray[p2[1]:p1[1], p2[0]:p1[0]]
    blurred_image = cv.blur(cropped_image, (30,30))

    resized_image = cv.resize(blurred_image, (28,28), interpolation=cv.INTER_LANCZOS4)
    resized_image = cv.bitwise_not(resized_image)

    resized_image = resized_image.astype(np.int16)
    scale = 3.5
    resized_image = resized_image * scale
    resized_image = np.clip(resized_image, 0, 255)
    resized_image = resized_image.astype(np.uint8)

    #cv.imshow('test', resized_image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return resized_image

