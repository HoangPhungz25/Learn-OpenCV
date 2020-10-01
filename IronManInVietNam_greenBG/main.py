import cv2 as cv
import numpy as np

urlIronMan =  r"iron_cut.mp4"
urlFlyCamVN = r"hanoi_cut.mp4"

ironManVid = cv.VideoCapture(urlIronMan)
flyCamVNVid = cv.VideoCapture(urlFlyCamVN)
iron_fourcc = cv.VideoWriter_fourcc(*'XVID')
out_iron_gray = cv.VideoWriter("iron_gray.avi", iron_fourcc, 24.0, (1280, 720), False)
out_iron_threshold = cv.VideoWriter("iron_threshold.avi", iron_fourcc, 24.0, (1280, 720), False)
out_iron_final = cv.VideoWriter("iron_final5.avi", iron_fourcc, 30, (1280, 720), True)

ret, firstImg_iron = ironManVid.read()
ret, firstImg_fly = flyCamVNVid.read()
row_iron, collum_iron, channel_iron = firstImg_iron.shape
row_fly, collum_fly, channel_fly = firstImg_fly.shape

ironManFPS = ironManVid.get(cv.CAP_PROP_FPS)
flyCamVNFPS = flyCamVNVid.get(cv.CAP_PROP_FPS)

contrast_kernel = np.ones((5,5), np.float32)/18*-1
# contrast_kernel = contrast_kernel*-1
contrast_kernel[2][2]=3

while True:

    ret, imgIron = ironManVid.read()
    ret, imgFlyCam = flyCamVNVid.read()

    # imgIron = cv.addWeighted(imgIron, 0.7, imgIron, 0.3, -60)
    #
    # imgIron = cv.filter2D(imgIron, -1, contrast_kernel)

    mask_all = cv.inRange(imgIron, (0,120,0),(140,255,140))

    cv.imshow("x", mask_all)

    #
    # g = imgIron[:,:,1]
    # r = imgIron[:,:,2]
    #
    #
    # # blur = cv.GaussianBlur(g, (1,1), 10)
    # ret, mask = cv.threshold(g, 200, 255, cv.THRESH_TOZERO)
    # ret, mask2 = cv.threshold(mask, 150, 255, cv.THRESH_BINARY)
    # mask_inv = cv.bitwise_not(mask2)
    #
    # ret, mask_r = cv.threshold(r, 200, 255, cv.THRESH_BINARY)
    #
    # mask_all = cv.add(mask_r, mask_inv)
    mask_all_inv = cv.bitwise_not(mask_all)
    #
    imgFlyCam_backout = cv.bitwise_and(imgFlyCam, imgFlyCam, mask = mask_all)
    iron_fg = cv.bitwise_and(imgIron, imgIron, mask = mask_all_inv)
    #
    iron_final = cv.add(imgFlyCam_backout, iron_fg)
    #
    # cv.imshow("ironman_green", g)
    cv.imshow("ironman_mask_inv", mask_all_inv)
    cv.imshow("ironman_fg", iron_fg)
    # cv.imshow("ironman_final", iron_final)
    cv.imshow("mask red", iron_final)
    #
    # out_iron_gray.write(g)
    # out_iron_threshold.write(mask_inv)

    out_iron_final.write(iron_final)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break;

ironManVid.release()
flyCamVNVid.release()
out_iron_gray.release()
out_iron_threshold.release()
out_iron_final.release()
cv.destroyAllWindows()






