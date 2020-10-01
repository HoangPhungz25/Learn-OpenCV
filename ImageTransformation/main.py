import cv2 as cv
import numpy as np
import copy

def add_one_layer(padiing):
    # add one layer
    img2 = cv.resize(img, (collum+padiing, row+padiing), interpolation=cv.INTER_LINEAR)
    img2_crop = img2[padiing//2:row+padiing//2, padiing//2:collum+padiing//2, ]

    mask = cv.inRange(img2_crop, (0,120,0),(140,255,140))

    mask_all = cv.bitwise_not(mask)

    img_bo = cv.bitwise_and(img2_crop, img2_crop, mask = mask_all)

    return cv.addWeighted(img_cp, 1,  img_bo, 0.33, 8)


# main
# iron = cv.VideoCapture(r"iron.mp4")
iron = cv.VideoCapture(r"duke_hien_body.mp4")
_, img = iron.read()
row, collum, channel = img.shape

out_fourcc = cv.VideoWriter_fourcc(*'XVID')
writer = cv.VideoWriter("out.avi", out_fourcc, 30, (collum, row), True)

while True:
    _, img = iron.read()
   #  img_bg_mask = cv.inRange(img, (0,120,0),(140,255,140)) # green bg

    img_blur = cv.GaussianBlur(img, (5,5), 5)
    img_bg_mask = cv.inRange(img_blur, (150,120,80),(255,255,255)) # blue bg

    img_fg_mask = cv.bitwise_not(img_bg_mask)
    img = cv.bitwise_and(img, img, mask = img_fg_mask)
    img = img//2
    
    img_cp = copy.deepcopy(img)
    img_cp = add_one_layer(30)
    img_cp = add_one_layer(70)
    img_cp = add_one_layer(130)

    cv.imshow("bg", img_bg_mask)

    # cv.imshow("img final", img_cp)
    writer.write(img_cp)

    k = cv.waitKey(1) & 0xFF
    if k==ord('q'): break

iron.release()
writer.release()
cv.destroyAllWindows()