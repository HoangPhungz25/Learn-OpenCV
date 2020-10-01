import cv2 as cv
import numpy as np


def enrode(img):
    r, c, ch = img.shape
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img_tr = cv.threshold(img_gray, 45, 255, cv.THRESH_BINARY)
    img_tr = cv.bitwise_not(img_tr)
    kernel = np.ones((15, 15), np.uint8)
    img_erode = cv.erode(img_tr, kernel, iterations=1)
    img_dilation = cv.dilate(img_tr, kernel, iterations=1)

    cv.namedWindow("x", cv.WINDOW_NORMAL)
    cv.resizeWindow("x", c // 2, r // 2)
    cv.namedWindow("x2", cv.WINDOW_NORMAL)
    cv.resizeWindow("x2", c // 2, r // 2)
    cv.namedWindow("x3", cv.WINDOW_NORMAL)
    cv.resizeWindow("x3", c // 2, r // 2)
    cv.imshow("x", img_tr)
    cv.imshow("x2", img_erode)
    cv.imshow("x3", img_dilation)


def close(img):
    kernel = np.ones((8, 8), np.uint8)
    img_nav = cv.bitwise_not(img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_close = cv.morphologyEx(img_nav, cv.MORPH_GRADIENT, kernel)
    cv.imshow("x", img_close)


def gradient(img):
    img_gradient = cv.Laplacian(img, -1, ksize=3)
    img_sobelx = cv.Sobel(img, -1, 1, 0, ksize=3)
    img_sobely = cv.Sobel(img, -1, 0, 1, ksize=3)

    cv.imshow("X", img_gradient)
    cv.imshow("sx", img_sobelx)
    cv.imshow("sy", img_sobely)


def gradientnumLayer4f(img):
    img_gradientnumLayer4 = cv.Sobel(img, cv.CV_numLayer4F, 1, 0, ksize=3)
    img_gradientnumLayer4Abs = np.absolute(img_gradientnumLayer4)
    img_gradient8u = np.uint8(img_gradientnumLayer4Abs)
    cv.imshow("gra_numLayer4", img_gradient8u)


def cannyEdge(img):
    canny = cv.Canny(img, 180, 255)

    img = cv.GaussianBlur(img, (5, 5), 0)

    img_sobel_xy = cv.Sobel(img, cv.CV_numLayer4F, 1, 0, ksize=1)
    img_sobel_xy = cv.Sobel(img_sobel_xy, cv.CV_numLayer4F, 1, 0, ksize=1)

    cv.imshow("xy", img_sobel_xy)
    cv.imshow("canny", canny)

def paramid(img):

    img_lower = cv.pyrDown(img)
    img_higher = cv.pyrUp(img)

    cv.imshow("or", img)
    cv.imshow("x", img_higher)
    cv.imshow("y", img_lower)

def blendImages(img1, img2):
    
    numLayer = 7;

    cv.imshow("origin",img1)
    cv.imshow("origin2",img2)


    G1 = img1.copy()
    G2 = img2.copy()
    gpA = [G1]
    gpB = [G2]

    for i in range(numLayer):
        G1 = cv.pyrDown(G1)
        G2 = cv.pyrDown(G2)
        gpA.append(G1)
        gpB.append(G2)

    lpA = [gpA[numLayer]]
    lpB = [gpB[numLayer]]
    for i in range(numLayer,0,-1):
        GeA = cv.pyrUp(gpA[i])
        GeB = cv.pyrUp(gpB[i])


        r, c, ch = gpA[i-1].shape
        la = cv.subtract(gpA[i-1], GeA[0:r, 0:c])
        lpA.append(la)

        r, c, ch = gpB[i-1].shape
        lb = cv.subtract(gpB[i-1], GeB[0:r, 0:c])
        lpB.append(lb)

    # joint left half and right half of each image
    LS = []
    for i in range(numLayer+1):
        la = lpA[i]
        lb = lpB[i]
        r, c, ch = lpA[i].shape
        ls = np.hstack((la[:,0:c//2],lb[:,c//2:]))
        LS.append(ls)

    l_ = LS[0]
    for i in range(1,numLayer+1):
        r,c,ch = LS[i].shape
        l_ = cv.pyrUp(l_)

        lCut = l_[0:r,0:c]
        l_ = cv.add(lCut, LS[i])
    l_[:,:,0] = cv.add(l_[:,:,0], 30)
    l_[:,:,1] = cv.subtract(l_[:,:,1], 25)
    l_[:,:,2] = cv.subtract(l_[:,:,2], 40)

    contrast_kernel = np.ones((5,5), np.uint8)*-1/24
    contrast_kernel[2,2] = 2
    l_ = cv.filter2D(l_, -1, contrast_kernel)

    cv.imwrite("blended_Robert_David.png", l_)
    cv.imshow("blended", l_)



img = cv.imread(r"duke.jpg")
img2 = cv.imread(r"close.png")
img3 = cv.imread(r"gradients.jpg")
img4 = cv.imread(r"gradients2.jpg")
img5 = cv.imread(r"messi_canny.jpg")

apple = cv.imread(r"orange.png")
orange = cv.imread(r"apple.png")

robert = cv.imread(r"robert.png")
david = cv.imread(r"david.png")

# enrode(img)
# close(img2)
# gradient(img2)
# gradientnumLayer4f(img2)
# cannyEdge(img5)
# paramid(img)
blendImages(apple, orange)
blendImages(robert, david)

cv.waitKey(0)
cv.destroyAllWindows()
