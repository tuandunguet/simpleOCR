import cv2
import numpy as np
import argparse

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
args = parser.parse_args()

# load image as grayscale
img = cv2.imread(args.input_file)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

resize_ratio = 500 / img.shape[0]
img = opencv_resize(img, resize_ratio)

# threshold 
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# get bounds of white pixels
white = np.where(thresh==255)
xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
print(xmin,xmax,ymin,ymax)

# crop the gray image at the bounds
crop = gray[ymin:ymax, xmin:xmax]
hh, ww = crop.shape

# do adaptive thresholding
thresh2 = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1.1)

# apply morphology
kernel = np.ones((1,7), np.uint8)
morph = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((5,5), np.uint8)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# invert
morph = 255 - morph

# get contours (presumably just one) and its bounding box
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x,y,w,h = cv2.boundingRect(cntr)

# draw bounding box on input
bbox = img.copy()
cv2.rectangle(bbox, (x+xmin, y+ymin), (x+xmin+w, y+ymin+h), (0,0,255), 1)

# test if contour touches sides of image
if x == 0 or y == 0 or x+w == ww or y+h == hh:
    print('region touches the sides')
else:
    print('region does not touch the sides')

# save resulting masked image
# cv2.imwrite('streak_thresh.png', thresh)
# cv2.imwrite('streak_crop.png', crop)
# cv2.imwrite('streak_bbox.png', bbox)

# display result
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.imshow("crop", crop)
cv2.waitKey(0)
cv2.imshow("thresh2", thresh2)
cv2.waitKey(0)
cv2.imshow("morph", morph)
cv2.waitKey(0)
cv2.imshow("bbox", bbox)
cv2.waitKey(0)
cv2.destroyAllWindows()