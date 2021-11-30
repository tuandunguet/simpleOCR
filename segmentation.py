import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from utils import plot_rgb, plot_gray, show_plot

debug=False
report=False

# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def getContourLength(contour):
  return cv2.arcLength(contour, False)

def get_receipt_contour(contours):    
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx

#######################################################
def getDistance(a,b):
  return np.linalg.norm(a - b)

def getOrderPoints(points):
    '''
    Divide points into 2 part: left and right
    sort the points based on x-coordinates
    '''
    sortArgX = np.argsort(points[:,0]) 
    left = np.array([points[x] for x in sortArgX[0:2]])
    right = np.array([points[x] for x in sortArgX[2:4]])
    # point with bigger y is bottomLeft and vice versa
    bottomLeft = left[np.argmax(left[:,1])]
    topLeft = left[np.argmin(left[:,1])]
    # point that is farther from the topLeft is bottomRight
    if getDistance(topLeft, right[0]) > getDistance(topLeft, right[1]):
        bottomRight = right[0]
        topRight = right[1]
    else:
        bottomRight = right[1]
        topRight = right[0]
    return (topLeft, topRight, bottomRight, bottomLeft)

def getIntersection(line1, line2):
    #lines are of the form (rho, theta)
    if debug:
        print(line1)
        print(line2)
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[math.cos(theta1), math.sin(theta1)],
                [math.cos(theta2), math.sin(theta2)]])
    B = np.array([rho1, rho2]) 
    if debug:
        print(A)
        print(B)
    #return form: np.array([x, y]), may raise exception
    result = np.linalg.solve(A, B) 
    return result

def getBoundaryIntersections(line, img):
    rho = line[0]
    theta = line[1]
    if theta >= math.pi / 2:
        newTheta = theta - math.pi / 2
    else:
        newTheta = theta + math.pi / 2
    height = img.shape[0]
    width = img.shape[1]
    leftBound = (0, 0)
    rightBound = (width, 0)
    topBound = (0, math.pi / 2)
    bottomBound = (height, math.pi / 2)
    bounds = (leftBound, rightBound, topBound, bottomBound)
    intersections = list()
    for bound in bounds:
        try:
            intersection = getIntersection(line, bound)
        except np.linalg.linalg.LinAlgError:
            continue
        else:
            intersections.append(intersection)
    rhos = [getRho(newTheta)(point) for point in intersections]
    intersections = np.array(intersections)
    numPoints = len(intersections)
    if numPoints == 4:
        return list(intersections[np.argsort(rhos)][1:3])
    elif numPoints == 2:
        return list(intersections)
    else:
        raise Exception("Error in GetBoundaryIntersections: Not enough points")

# Geting `rho` of a line that goes through `point` with angle `theta`
def getRho(theta):
    def result(point):
        #point is of the form (x, y)
        return point[0] * math.cos(theta) + point[1] * math.sin(theta)
    return result

def checkSimilarRho(line, correctLine, img, rhoErr = 20):
    rho, theta = line
    intersections = getBoundaryIntersections(correctLine, img)
    rhos = [getRho(theta)(intersection) for intersection in intersections]
    if rho < max(rhos) + rhoErr and rho > min(rhos) - rhoErr:
        return True
    return False

def getParallel(line, point):
    rho, theta = line
    return getRho(theta)(point), theta

#####################################################################

#------------Some constants----------------
#Minimum angle different needed between 2 lines for them to be detect as separated
thetaErr = math.pi / 6
#Minimum space between 2 lines for them to be detected as separated
rhoErr = 20 

#------------Helper functions--------------
def getLength(contour):
    return cv2.arcLength(contour, False)

def checkSimilarAngle(theta1, theta2):
    if theta1 <= thetaErr / 2 and theta2 >= math.pi - thetaErr / 2:
        return True#, (theta1, theta2)
    elif theta2 <= thetaErr / 2 and theta1 >= math.pi - thetaErr / 2:
        return True#, (theta2, theta1)
    elif abs(theta1 - theta2) < thetaErr:
        return True#, (theta1, theta2)
    else:
        return False#, None

# get missing edges in case only three edges are detected
# return [lonely edges, pair edges]
def getMissingEdges(correctLines):
    if checkSimilarAngle(correctLines[0][1], correctLines[1][1]):
        return [correctLines[2], correctLines[0], correctLines[1]]
    elif checkSimilarAngle(correctLines[0][1], correctLines[2][1]):
        return [correctLines[1], correctLines[0], correctLines[2]]
    else:
        return correctLines

# Draw lines into img, `diagonal` is the diagonal of img
def drawLines(lines, img, diagonal, thick = 3):
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + math.ceil(diagonal) * (-b))
        y1 = int(y0 + math.ceil(diagonal) * a)
        x2 = int(x0 - math.ceil(diagonal) * (-b))
        y2 = int(y0 - math.ceil(diagonal) * a)
        cv2.line(img, (x1,y1), (x2, y2), (255,255,255), thick)
    if debug == True:
        print(lines)

# resize image for faster processing
def resizeImg(img):
    width = img.shape[1]
    height = img.shape[0]
    if width > 700:
        ratio = 500 / width
        resized = cv2.resize(img, None, fx = ratio, fy = ratio, interpolation = cv2.INTER_LINEAR) 
        newWidth = 500
        newHeight = int(height * ratio)
        return resized, ratio, newWidth, newHeight
    else:
        if width < 300:
            print("Warning: Image is too small")
        return img, 1, width, height

#################################################################

def receipt_crop(image, showProgress = False):
    try:
        originalImg = image.copy()

        image, ratio, newWidth, newHeight = resizeImg(image)

        height, width = image.shape[0], image.shape[1]

        # Convert to grayscale for further processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get rid of noise with Gaussian Blur filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect white regions
            # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            # dilated = cv2.dilate(blurred, rectKernel)

        # find edge
        edged = cv2.Canny(blurred, 100, 200, apertureSize=3)

        if (showProgress):
            plot_gray(edged)

        # Detect all contours in Canny-edged image
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)

        if (showProgress):
            plot_rgb(image_with_contours)

        # Get longest contour
        longestContour = max(contours, key = getContourLength)
        image_with_longest_contour = cv2.drawContours(image.copy(), [longestContour], -1, (0,255,0), 3)

        if (showProgress):
            plot_rgb(image_with_longest_contour)

        black = np.zeros((height, width), "uint8")
        cv2.drawContours(black, [longestContour], -1, (255,255,255), 1)

        if (showProgress):
            plot_rgb(black)

        lines = cv2.HoughLines(black,1,np.pi/180, 50)

        correctLines = list()
        for line in lines:
            if len(correctLines) == 4:
                break
            rho,theta = line[0]
            isNew = True
            numSimilar = 0
            for l in correctLines:
                correctTheta = l[1]
                if checkSimilarAngle(theta, correctTheta):
                    numSimilar += 1
                    if numSimilar == 2:
                        isNew = False
                        break
                    if checkSimilarRho(line[0], l, image, rhoErr):
                        isNew = False
                        break
            if isNew:
                correctLines.append([rho, theta])
            else:
                continue
    
        for line in correctLines:
            line[0] = line[0] / ratio
            numLines = len(correctLines)
            if numLines < 3:
                raise Exception("Error: Receipt does not have enough edges to detect!")
            elif numLines == 3:
                correctLines = getMissingEdges(correctLines)
                rho, theta = correctLines[0]
                intersections = getBoundaryIntersections(correctLines[1], originalImg) + getBoundaryIntersections(correctLines[2], originalImg)
                intersections.sort(key = getRho(theta))
                if abs(getRho(theta)(intersections[1]) - rho) > abs(getRho(theta)(intersections[2]) - rho):
                    newLine = getParallel(correctLines[0], intersections[1])
                else:
                    newLine = getParallel(correctLines[0], intersections[2])
                correctLines.append(newLine)
    
        corners = list()
        for i in range(4):
            for j in range(i + 1, 4):
                if checkSimilarAngle(correctLines[i][1], correctLines[j][1]):
                    continue
                try:
                    intersection = getIntersection(correctLines[i], correctLines[j])
                except np.linalg.linalg.LinAlgError:
                    continue
                else:
                    corners.append(intersection)
        if len(corners) != 4:
            raise Exception('Error: Cannot get correct corners')
        topLeft, topRight, bottomRight, bottomLeft = getOrderPoints(np.array(corners, dtype = "float32"))
        oldCorners = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype = "float32")
        #Compute new width and height
        newWidth = max(getDistance(topLeft, topRight), getDistance(bottomLeft, bottomRight))
        newHeight = max(getDistance(topLeft, bottomLeft), getDistance(topRight, bottomRight))
        #Compute 4 new corners
        newCorners = np.array([
            [0, 0],
            [newWidth - 1, 0],
            [newWidth - 1, newHeight -1],
            [0, newHeight -1]], dtype = "float32")
        #Compute transformation matrix
        transMat = cv2.getPerspectiveTransform(oldCorners, newCorners)
        #Transform
        resultImage = cv2.warpPerspective(originalImg, transMat, (int(newWidth), int(newHeight)))

        if (showProgress):
            plot_rgb(resultImage)
        
        if (showProgress):
            show_plot()

    # return original image if error happen
    except Exception as e:
        return image

    return resultImage


#################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    args = parser.parse_args()

    image = cv2.imread(args.input_file)

    receipt_cropped_img = receipt_crop(image)

    cv2.imwrite(receipt_cropped_img, args.output_file)