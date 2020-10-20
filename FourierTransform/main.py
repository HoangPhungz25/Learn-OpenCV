import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def drawFrequenceDomain(url):

    # Find fourier transform image of a picture form brightness image then draw it
    img = cv.imread(url)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_fft = cv.dft(np.float32(img_gray), flags=cv.DFT_COMPLEX_OUTPUT)
    fft_shift = np.fft.fftshift(img_fft)

    magnitude_spectrum = 20*np.log(cv.magnitude(fft_shift[:,:,0], fft_shift[:,:,1]))

    # reverse = cv.idft(img_fft)
    # reverse = cv.magnitude(reverse[:,:,0], reverse[:,:,1])

    r, c, ch = img.shape
    cr,cc = r//2, c//2

    mask = np.ones((r,c,2), np.uint8)
    padding = 10
    # mask[:,cc-padding:cc+padding]=0
    mask[cr-padding:cr+padding,:]=0


    fft_shift = fft_shift*mask
    mask_frequence = cv.magnitude(fft_shift[:,:,0],fft_shift[:,:,1])

    fft_shift = np.fft.fftshift(fft_shift)
    img_back = cv.idft(fft_shift)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])


    plt.subplot(231), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(img_back, cmap='gray')
    plt.title('Reverse Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(mask_frequence, cmap='gray')
    plt.title('Mask frequence'), plt.xticks([]), plt.yticks([])
    plt.show()


img_url = r"slap.jpg"
drawFrequenceDomain(img_url)
cv.waitKey(0)
cv.destroyAllWindows()
