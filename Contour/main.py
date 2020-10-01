import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def mdrawContour(img):

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img_gray, 45, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    red = 0;
    b = 0
    for x in range(len(contours)):

        #draw contours with different color
        cv.drawContours(img, contours, x, (b, 200, red), 1)
        red = red+20
        b +=10

        #plot the contour point
        plt.plot(x[:,:,0], x[:,:,1], 'bo')
        plt.plot(x[:,:,0], x[:,:,1], 'r+')

    ct0 = contours[37]
    mo0 = cv.moments(ct0)
    ctArea = cv.contourArea(ct0)


    # draw contour with epsilon
    perimeter = cv.arcLength(ct0, True)
    ep = 0.004*perimeter
    apx = cv.approxPolyDP(ct0, ep, True)
    cv.drawContours(img,[apx], -1, (0,0,255), 5)


    # draw a circle at centroid of the body which has the largest contour
    cx = int(mo0['m10']/mo0['m00'])
    cy = int(mo0['m01']/mo0['m00'])

    cv.circle(img, (cx, cy), 100, (0, 255, 0), 2)
    plt.show()

    cv.imshow("g", img_gray)
    cv.imshow("t", thresh)
    cv.imshow("x", img)

def convexHull(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thres = cv.threshold(img_gray, 70, 255, cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for ct in contours:

        # draw convex hull
        hull = cv.convexHull(ct, clockwise=True, returnPoints=True)
        if cv.isContourConvex(ct):
            cv.drawContours(img, [hull], -1, (0,255,255), 1)
        else:
            cv.drawContours(img, [hull], -1, (255, 0, 0), 1)

        # draw bounding rect
        x,y,w,h = cv.boundingRect(ct)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # draw rotated bounding rect
        for i in range(15):
            rect = cv.minAreaRect(ct)
            rect2 = list(rect)
            rect2[2]+=i*12
            rect = tuple(rect2)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img, [box], -1, (100,20+i*15,35+i*20), 2, cv.LINE_AA)

    cv.imshow("x", img)
    cv.imshow("thres", thres)
    cv.imwrite("magicHand.jpg", img)

def convexHull_MagicHand_Effect(video, thresholeb, thresholeg, thresholer, thresType, b, bPadding, g, gPadding, re, rPadding, thicknes, numOfRect, degreePadding):
    cap = cv.VideoCapture(video)
    _, img = cap.read()
    r,c,ch = img.shape
    out_fourcc = cv.VideoWriter_fourcc(*'XVID')
    outWriter = cv.VideoWriter("out.avi", out_fourcc, 30.0, (c, r), True)

    kernel = np.ones((25,25),np.float32)/(25*25)

    while True:
        _, img = cap.read()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.filter2D(img, -1, kernel)
        blur = cv.GaussianBlur(blur, (81,81), 50)
        blur = cv.GaussianBlur(blur, (81,81), 100)
        blur = cv.GaussianBlur(blur, (81,81), 100)

        # _, thres = cv.threshold(blur, thresholeb, 255, thresType)
        thres = cv.inRange(blur, (thresholeb, thresholeg, thresholer), (255,255,255))

        contours, hierarchy = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for ct in contours:
            # cv.drawContours(img, contours, -1, (0,255,255), 2)

            # approximated contour
            arcLen = cv.arcLength(ct, True)
            ep = 0.01*arcLen
            approx = cv.approxPolyDP(ct, ep, True)
            cv.drawContours(img, [approx], -1, (0,255,255), 2)

            rect = cv.minAreaRect(approx)

            xCenter = np.int0(rect[0][0])
            yCenter = np.int0(rect[0][1])
            radius = np.int0(math.sqrt((np.int0(rect[1][0])**2+np.int0(rect[1][1]**2))))//2
            radius_triangle = radius//2-50

            # diagonalLine = math.sqrt(((rect[0][0]-rect[1][0])**2)+((rect[0][1]-rect[1][1])**2))
            # diagonalLine = np.int0(diagonalLine)
            #
            # xCenter = np.int0(rect[1][0] + diagonalLine//2)
            # yCenter = np.int0(rect[1][1] + diagonalLine//2)
            #
            # cv.circle(img, (np.int0(rect[0][0]),np.int0(rect[0][1])), 50, (0,0,255), 12)
            # # cv.circle(img, (np.int0(rect[1][0]),np.int0(rect[1][1])), 50, (0,0,0), 12)
            #
            #
            # diagonalLine +=50


            for i in range(numOfRect): #15

                rect2 = list(rect)
                rect2[2] += i*degreePadding #0.5
                rect = tuple(rect2)

                box = cv.boxPoints(rect)
                box = np.int0(box)

                cv.drawContours(img, [box], -1, (b+i*bPadding, g+i*gPadding, re+i*rPadding), thicknes)
                cv.circle(img, (xCenter, yCenter), abs(radius//2-i*2), (b+i*bPadding, g+i*gPadding, re+i*rPadding), 2)
                cv.circle(img, (xCenter, yCenter), radius+50+i*5, (b+i*bPadding, g+i*gPadding,  re+i*rPadding), 2)
                # draw triangle
                radius_triangle = (radius_triangle-5)
                p1_triangle = (xCenter-radius_triangle, yCenter)
                p2_triangle = (xCenter+np.int0(np.sin(np.pi/6)*(radius_triangle)), yCenter-np.int0(np.cos(np.pi/6)*(radius_triangle)))
                p3_triangle = (xCenter+np.int0(np.sin(np.pi/6)*(radius_triangle)), yCenter+np.int0(np.cos(np.pi/6)*(radius_triangle)))
                cv.line(img, p1_triangle, p2_triangle, (b+180-i*bPadding,g+180,re+120+i*rPadding), 2)
                cv.line(img, p2_triangle, p3_triangle, (b+180-i*bPadding,g+180,re+120+i*rPadding), 2)
                cv.line(img, p3_triangle, p1_triangle, (b+180-i*bPadding,g+180,re+120+i*rPadding), 2)





        cv.namedWindow("x", cv.WINDOW_NORMAL)
        cv.resizeWindow("x", c//2, r//2)
        cv.imshow("x", img)
        # cv.imshow("x", thres)
        outWriter.write(img)
        # cv.imshow("thres", thres)

        k = cv.waitKey(1)
        if k == ord('q'):
            break;
        elif k== ord(' '):
            time.sleep(1)

    cap.release()
    outWriter.release()

# main


img = cv.imread(r"duke.png")
img2 = cv.imread(r"hand.png")
# mdrawContour(img)
# convexHull(img2)
# convexHull_MagicHand_Effect(r"hand_vid.mp4", 40, cv.THRESH_BINARY, 100, 0, 25, 18, 35, 10, 2, 20, 0.6)
convexHull_MagicHand_Effect(r"doctorStrange.mp4", 140, 140, 140, cv.THRESH_BINARY, 50, 25, 35, 16, 25, 8, 2, 18, 0.6)

cv.waitKey(0)
cv.destroyAllWindows()
