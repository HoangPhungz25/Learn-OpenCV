import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plotHistogram(img):
    img = cv.imread(img)
    color = ('b','g','r')
    for i, col in enumerate(color):
        his = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(his, color = col)
    plt.show()

# main
img = "img.png"
img2 = r"dune.png"
# plotHistogram(img)
# plotHistogram(img2)
R = np.ones([15,30])
arr = np.int0(np.ones([600,600])*np.random.randint(15))
arr2 = np.int0(np.ones([600,600])*np.random.randint(25))
ax = arr.ravel()
bx = arr2.ravel()
H = R[ax[:], bx[:]]
H
x = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
cv.waitKey(0)
cv.destroyAllWindows()
