import numpy as np
import cv2

import argparse

def createDetector():
    detector = cv2.ORB_create(nfeatures=2000)
    return detector

def getFeatures(img):
    detector = createDetector()
    kps, descs = detector.detectAndCompute(img, None)
    return kps, descs, img.shape[:2][::-1]

def detectFeatures(img, train_features):
    train_kps, train_descs, shape = train_features
    # get features from input image
    kps, descs, _ = getFeatures(img)
    # check if keypoints are extracted
    if not kps:
        return None
    # now we need to find matching keypoints in two sets of descriptors (from sample image, and from current image)
    # knnMatch uses k-nearest neighbors algorithm for that
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(train_descs, descs, k=2)
    good = []
    # apply ratio test to matches of each keypoint
    # idea is if train KP have a matching KP on image, it will be much closer than next closest non-matching KP,
    # otherwise, all KPs will be almost equally far
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    # stop if we didn't find enough matching keypoints
    if len(good) < 0.1 * len(train_kps):
        return None
    # estimate a transformation matrix which maps keypoints from train image coordinates to sample image
    src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if m is not None:
        # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
        scene_points = cv2.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1), (shape[1] - 1, shape[0] - 1), (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)
        rect = cv2.minAreaRect(scene_points)
        # check resulting rect ratio knowing we have almost square train image
        if rect[1][1] > 0 and 0.8 < (rect[1][0] / rect[1][1]) < 1.2:
            return rect
    return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--logo_file", "-l", type=str, help="Path to logo image")
	parser.add_argument("--image_file", "-i", type=str, help="Path to image")
	args = parser.parse_args()

	logo = cv2.imread(args.logo_file, cv2.COLOR_BGR2GRAY)
	img = cv2.imread(args.image_file, cv2.COLOR_BGR2GRAY)

	# # get train features
	# train_features = getFeatures(logo)
	# # detect features on test image
	# region = detectFeatures(img, train_features)
	# if region is not None:
		# print('logo is containned')
		# # draw rotated bounding box
		# box = cv2.boxPoints(region)
		# box = np.int0(box)
		# cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
	# # display the image
	# cv2.imshow("Preview", img)
	# cv2.waitKey(0)

	# Read the query image as query_img
	# and train image This query image
	# is what you need to find in train image
	# Save it in the same directory
	# with the name image.jpg 
	query_img = img
	train_img = logo
	
	# Convert it to grayscale
	query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
	train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
	
	# Initialize the ORB detector algorithm
	orb = cv2.ORB_create()
	
	# Now detect the keypoints and compute
	# the descriptors for the query image
	# and train image
	queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
	trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
	
	# Initialize the Matcher for matching
	# the keypoints and then match the
	# keypoints
	matcher = cv2.BFMatcher()
	matches = matcher.match(queryDescriptors,trainDescriptors)
	
	# draw the matches to the final image
	# containing both the images the drawMatches()
	# function takes both images and keypoints
	# and outputs the matched query image with
	# its train image
	final_img = cv2.drawMatches(query_img, queryKeypoints,
	train_img, trainKeypoints, matches[:20],None)
	
	final_img = cv2.resize(final_img, (1000,700))
	
	# Show the final image
	cv2.imshow("Matches", final_img)
	cv2.waitKey(0)
