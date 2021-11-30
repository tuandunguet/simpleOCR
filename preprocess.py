import argparse
import numpy as np
import cv2
from numpy.core.fromnumeric import resize

from segmentation import receipt_crop

from utils import plot_rgb, plot_gray, show_plot

# -----------------------------------------

# resize to 300 dpi
def resize(img):
  img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)  #Inter Cubic 
  return img

# Increase Brightness
def checkDayOrNight(img, thrshld):
    is_light = np.mean(img) > thrshld
    return 0 if is_light else 1 # 0 --> light and 1 -->dark
def increaseBrightness(img):
  alpha = 1 
  beta = 40
  img=cv2.addWeighted(img, alpha,np.zeros(img.shape,img.dtype), 0 ,beta)
  return img


# Noise reduction
def noise_remove(img):
  kernel = np.ones((1, 1), np.uint8)
  img = cv2.dilate(img, kernel, iterations=1)
  img = cv2.erode(img, kernel, iterations=1)
  return img

# -------------------------------------------

def preprocess(image, showProgress=False):
    # receipt segmentation
    receipt_cropped_img = receipt_crop(image)

    # resize to 300 dpi
    resized = resize(receipt_cropped_img)

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    if (showProgress):
        plot_rgb(gray)

    # increase brightness if too dark
    if checkDayOrNight(gray, 130) == 1:
        gray = increaseBrightness(gray)

    if (showProgress):
        plot_rgb(gray)

    # Contrast equalization
    # equalized = cv2.equalizeHist(gray)

    # if (showProgress):
        # plot_rgb(equalized)
    
    # Get rid of noise with Gaussian Blur filter

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if (showProgress):
        plot_gray(blurred)

    # Binarization using adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 9)

    if (showProgress):
        plot_gray(thresh)

    noise_removed = noise_remove(thresh)

    if (showProgress):
        plot_gray(noise_removed)

    # thinning
    # kernel = np.ones((3,3),np.uint8)
    # erosion = cv2.erode(thresh,kernel,iterations = 1)

    result = noise_removed

    if (showProgress):
        show_plot()

    return result

#################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    args = parser.parse_args()

    image = cv2.imread(args.input_file)

    preprocessd_image = preprocess(image, showProgress=True)
