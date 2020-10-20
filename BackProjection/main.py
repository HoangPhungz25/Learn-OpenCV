import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def testHisto(url):
    img = cv.imread(url)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    histo = cv.calcHist([img_hsv],[0,1], None, [180, 256],[0, 180, 0, 256])

    cv.imshow("origin", img)
    cv.imshow("hsv", img_hsv)
    cv.imshow("histo 2D", histo)

    plt.imshow(histo, interpolation='nearest')
    plt.show()


def backProjection(img_url):
    # Use back projection to find interested object

    img = cv.imread(img_url)
    r, c, ch = img.shape

    roi = img[200:500, 200:500]

    #convert to hsv color space
    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #calculate histogram
    roi_histo = cv.calcHist([roi_hsv],[0,1], None, [180, 256],[0, 180, 0, 256])
    # cv.imshow("roi histo1", roi_histo)
    #
    # cv.normalize(roi_histo, roi_histo, 0, 255, cv.NORM_MINMAX)
    # cv.imshow("roi histo2", roi_histo)

    img_histo = cv.calcHist([img_hsv], [0,1], None, [180, 256], [0, 180, 0, 256])

    R = roi_histo/img_histo

    h,s,v = cv.split(img)
    H = R[h.ravel(), s.ravel()]
    H = np.minimum(H, 1)

    dics = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    H = cv.filter2D(H, -1, dics)

    H = np.uint8(H)
    H = np.reshape(H, [r,c])

    cv.normalize(H, H, 0, 255, cv.NORM_MINMAX)

    # ret, thres = cv.threshold(H, 50, 255, cv.THRESH_BINARY)
    # thres = cv.merge((thres, thres, thres))

    H = cv.merge((H, H, H))


    final = cv.bitwise_and(img, H)


    cv.namedWindow("out", cv.WINDOW_NORMAL)
    cv.resizeWindow("out", c // 2, r // 2)

    cv.imshow("out", img)
    cv.imshow("roi", roi)
    cv.imshow("roi histo", roi_histo)
    cv.imshow("img histo", img_histo)
    cv.imshow("R", R)

    cv.namedWindow("outH", cv.WINDOW_NORMAL)
    cv.resizeWindow("outH", c // 2, r // 2)
    cv.imshow("outH", H)
    cv.imshow("final", final)





# main
imgUrl = r"Duke.jpg"
imgUrl_test = r"test_histo.png"
backProjection(imgUrl)
# testHisto(imgUrl_test)
cv.waitKey(0)
cv.destroyAllWindows()
