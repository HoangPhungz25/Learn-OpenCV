import numpy as np
import cv2 as cv

#   var
padding_percentage = 2
rect_width_percentage = 8
numOfRect = 3


video = cv.VideoCapture(r"cam.mp4")
ret, firstImge = video.read()
height, width,ch = firstImge.shape
height=height//3
width=width//3

padding_creen = width*padding_percentage//100
rectWidth = width*rect_width_percentage//100


rectVelocity = 5

currentRect_origin = [[padding_creen,0],[(width-2*padding_creen)//2-rectWidth//2,0],[width-padding_creen-rectWidth,0]]
currentRect = [[padding_creen,0],[(width-2*padding_creen)//2-rectWidth//2,0],[width-padding_creen-rectWidth,0]]


rectColor = (0,255,255)
faceColor = (255,0,0)


print(cv.data.haarcascades)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
body_cascade = cv.CascadeClassifier("haarcascade_fullbody.xml")



def createWindow(name,w,h):

    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w,h)


#   main


while True:
    ret, img = video.read()

    img = cv.resize(img, (width, height))

    createWindow("origin", width, height)

    #update new position of rect
    currentRect[0][1] += rectVelocity
    currentRect[1][1] += rectVelocity
    currentRect[2][1] += rectVelocity

    cv.rectangle(img, (currentRect[0][0], currentRect[0][1]), (currentRect[0][0]+rectWidth, currentRect[0][1]+rectWidth), rectColor, -1)
    cv.rectangle(img, (currentRect[1][0], currentRect[1][1]), (currentRect[1][0]+rectWidth, currentRect[1][1]+rectWidth), rectColor, -1)
    cv.rectangle(img, (currentRect[2][0], currentRect[2][1]), (currentRect[2][0]+rectWidth, currentRect[2][1]+rectWidth), rectColor, -1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2.0, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y),(x+w, y+h), faceColor, 5)
        rect_index = 0;
        for (recX, rectY) in currentRect:
            centerRect_x = recX+rectWidth//2
            centerRect_y = rectY+rectWidth//2

            if centerRect_x > x and centerRect_y > y and centerRect_x < x+w and centerRect_y < y+h:
                currentRect[rect_index] = np.copy(currentRect_origin[rect_index])


            rect_index+=1

    cv.imshow("origin", img)


    k = cv.waitKey(20)
    if k == ord('q'): break

