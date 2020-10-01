# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import numpy
import copy

urlHienMoving = r"C:\Users\PhungVanHoang\Videos\hienMoving.mp4"
urlTenet = r"C:\Users\PhungVanHoang\PycharmProjects\pythonProject\TENET_thunder_origin.mp4"
urlOutput = r"C:\Users\PhungVanHoang\PycharmProjects\pythonProject\outputThunder.avi"

urlDune = r"C:\Users\PhungVanHoang\Pictures\Background\0520-Dune-Timothee-Solo-Lede.jpg"
urlImage2 = r"C:\Users\PhungVanHoang\Pictures\Background\ATLAS-OF-PLACES-ANDREI-TARKOVSKY-STALKER-IMG-17.png"

urlLogo = r"C:\Users\PhungVanHoang\Pictures\Warner_Bros_logo.png"
urlStalker = r"C:\Users\PhungVanHoang\Pictures\Background\ATLAS-OF-PLACES-ANDREI-TARKOVSKY-STALKER-IMG-23.png"

def getWindow(row, collum, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, collum // 2, row // 2)


def cannyEdge(url):
    cap = cv2.VideoCapture(url)
    print(cap)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = (cv2.Canny(gray, 1, 100))

    h, w, num_color = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outGray = cv2.VideoWriter("outputGray.avi", fourcc, 24.0, (w, h), False)
    outThunder = cv2.VideoWriter("outputThunder.avi", fourcc, 24.0, (w, h), False)
    outFG = cv2.VideoWriter("outputFG.avi", fourcc, 24.0, (w, h), False)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgSub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        gray = numpy.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        edge = numpy.array(cv2.Canny(gray, 10, 30))
        fg = fgSub.apply(frame)
        # fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        effect = cv2.add(gray, edge)

        cv2.imshow("tenet edge", fg)

        outGray.write(gray)
        outThunder.write(effect)
        outFG.write(fg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    outGray.release()
    outThunder.release()
    outFG.release()

    cap.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(urlOutput)
    print(cap)
    while True:
        ret, frame = cap.read()

        cv2.imshow("output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


def ROI_cutImage(url):
    # cut a region of image then paste to onother

    img = cv2.imread(url)
    row, collum, color = img.shape

    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("window", collum // 2, row // 2)

    head = img[340:480, 1200:1350]
    img[200:340, 300:450] = head

    cv2.imshow("window", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_channel(url):
    img = cv2.imread(url)
    imgb = copy.deepcopy(img)
    imgg = copy.deepcopy(img)
    imgr = copy.deepcopy(img)

    imgb[:, :, 1] = 0
    imgb[:, :, 2] = 0

    imgg[:, :, 0] = 0
    imgg[:, :, 2] = 0

    imgr[:, :, 0] = 0
    imgr[:, :, 1] = 0

    row, collum, c = img.shape

    getWindow(row, collum, "b")
    getWindow(row, collum, "g")
    getWindow(row, collum, "r")

    cv2.imshow("b", imgb)
    cv2.imshow("g", imgg)
    cv2.imshow("r", imgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blend_2_image(url1, url2, ratio_number_1, ratio_number_2):
    # lend 2 images together with certain ratio

    img1 = cv2.imread(url1)
    img2 = cv2.imread(url2)

    cropped_width = 700
    cropped_height = 500

    croppedImg1 = img1[200:700, 900:2000]
    croppedImg2 = img2[200:700, 900:2000]

    # blended_img = cv2.addWeighted(croppedImg1, ratio_number_1, croppedImg2, ratio_number_2,0 )
    blended_img = [500, 1100]
    blended_img = cv2.add(croppedImg1 // ratio_number_1, croppedImg2 // ratio_number_2)
    cv2.imshow("Blended Image", blended_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def roi_none_retange(url1, url2):
    # bit-wise operations to add a logo to an image

    img1 = cv2.imread(url1)
    img2 = cv2.imread(url2)

    # get logo shape
    row, collum, channel = img2.shape

    # roi of img where we put the logo
    roi = img1[0:row, 0:collum]

    # create a mask of logo
    logo_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi_black_out = cv2.bitwise_and(roi, roi, mask = mask_inv)

    logo_fg = cv2.bitwise_and(img2, img2, mask = mask)

    roi_with_logo = cv2.add(roi_black_out, logo_fg)

    img1[0:row, 0:collum] = roi_with_logo


    getWindow(row, collum, "roi_black_out")
    cv2.imshow("roi_black_out", roi_black_out)

    getWindow(row, collum, "logo fg")
    cv2.imshow("logo fg", logo_fg)

    getWindow(row, collum, "roi with logo")
    cv2.imshow("roi with logo", roi_with_logo)

    getWindow(768,1366, "img with logo")
    cv2.imshow("img with logo", img1)




    cv2.waitKey(0)
    cv2.destroyAllWindows()


# main program
# not-use fun
# cannyEdge(urlTenet)
# ROI_cutImage(urlImage)
# color_channel(urlImage)
# blend_2_image(urlDune, urlImage2, 3, 2)
# roi_none_retange(urlStalker, urlLogo)