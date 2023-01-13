from __future__ import print_function
import cv2 as cv
import numpy as np

FLANN_INDEX_KDTREE = 1

def resize_image(img):
    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def subtract_background(bkg_img, img):
    fgbg = cv.createBackgroundSubtractorMOG2()
    fgbg.apply(bkg_img)
    fgmask = fgbg.apply(img)
    fgmask = cv.GaussianBlur(fgmask, (15, 15), 2)
    return cv.threshold(fgmask, 120, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

def get_object_mask(fgmask):
    contours = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        area = cv.contourArea(c)
        if area < 950:
            cv.drawContours(fgmask, [c], -1, (0, 0, 0), -1)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))

    open = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=5)

    return cv.morphologyEx(open, cv.MORPH_CLOSE, kernel, iterations=5)

def find_matching_keypoints(source_desriptors, dest_img):
    kp, des = sift_features.detectAndCompute(dest_img, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=150)
    search_params = dict(checks=1500)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(source_desriptors, des, k=2 )

    match_mask = [[0, 0] for j in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            match_mask[i] = [1, 0]

    return match_mask, matches, kp

def detect_object(object_to_detect_keypoints, object_to_detect_descriptors, img, all_img, draw, draw_title):
    match_mask, matches, kp = find_matching_keypoints(object_to_detect_descriptors, img)

    if (draw):
        object_keypoints_img = all_img.copy()
        cv.drawKeypoints(object_keypoints_img, object_to_detect_keypoints, 0, (0, 255, 0),
                                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        list_kp1 = []

        for _, (m, _) in enumerate(matches):
            img_idx = m.trainIdx

            (x1, y1) = kp[img_idx].pt

            list_kp1.append((x1, y1))
        matched_key_points = np.array(list_kp1)
        centroid = np.rint(np.sum(matched_key_points, axis=0)/len(matched_key_points)).astype(int)


        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=match_mask,
                           flags=cv.DrawMatchesFlags_DEFAULT)
        matches_img = cv.drawMatchesKnn(all_img, object_to_detect_keypoints, img, kp, matches, None,
                                        **draw_params)

        matches_img = cv.circle(matches_img, (centroid[0] + object_keypoints_img.shape[0], centroid[1]), 3, (0, 0, 255), 4)

        cv.imshow(draw_title, matches_img)
        cv.waitKey()

bkg_img = resize_image(cv.imread("./IMG_BKG.JPG"))[100:400, 80:400]

all_img = resize_image(cv.imread("./IMG_ALL.JPG"))[109:409, 77:397]

img_1 = resize_image(cv.imread("./IMG_1.JPG"))[100:400, 80:400]
img_2 = resize_image(cv.imread("./IMG_2.JPG"))[100:400, 80:400]
img_3 = resize_image(cv.imread("./IMG_3.JPG"))[100:400, 80:400]
img_4 = resize_image(cv.imread("./IMG_4.JPG"))[100:400, 80:400]
img_5 = resize_image(cv.imread("./IMG_5.JPG"))[100:400, 80:400]

cv.imshow("Background", bkg_img)

cv.waitKey()
cv.imshow("All", all_img)

cv.waitKey()

(thresh, fgmask) = subtract_background(bkg_img, all_img)

cv.imshow("Background Subtracted", fgmask)
cv.waitKey()

objects = get_object_mask(fgmask)

cv.imshow("objects", objects)
cv.waitKey()

contours2 = cv.findContours(objects, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

with_contours = all_img.copy()
cv.drawContours(with_contours, contours2[0], -1, (0, 255, 0), 3)
cv.imshow("with contours", with_contours)
cv.waitKey()

gum_mask = np.zeros((all_img.shape[0], all_img.shape[1], 1), np.uint8)
cv.drawContours(gum_mask, [contours2[0][0]], -1, (255), thickness=cv.FILLED)

scissors_mask = np.zeros((all_img.shape[0], all_img.shape[1], 1), np.uint8)
cv.drawContours(scissors_mask, [contours2[0][5]], -1, (255), thickness=cv.FILLED)

sunglasses_mask = np.zeros((all_img.shape[0], all_img.shape[1], 1), np.uint8)
cv.drawContours(sunglasses_mask, [contours2[0][2]], -1, (255), thickness=cv.FILLED)

mouse_mask = np.zeros((all_img.shape[0], all_img.shape[1], 1), np.uint8)
cv.drawContours(mouse_mask, [contours2[0][1]], -1, (255), thickness=cv.FILLED)

cv.imshow("gum_mask", gum_mask)
cv.waitKey()

cv.imshow("mouse_mask", mouse_mask)
cv.waitKey()

cv.imshow("scissors_mask", scissors_mask)
cv.waitKey()

cv.imshow("sunglasses_mask", sunglasses_mask)
cv.waitKey()

objects = cv.bitwise_and(all_img, all_img, mask=objects)
cv.imshow("Objects", objects)
cv.waitKey()

objects = cv.cvtColor(objects, cv.COLOR_BGR2GRAY)
sift_features = cv.SIFT_create()
mouse_keypoints, mouse_descriptors = sift_features.detectAndCompute(objects, mouse_mask)
sunglasses_keypoints, sunglasses_descriptors = sift_features.detectAndCompute(objects, sunglasses_mask)
gum_keypoints, gum_descriptors = sift_features.detectAndCompute(objects, gum_mask)
scissors_keypoints, scissors_descriptors = sift_features.detectAndCompute(objects, scissors_mask)

thresh, img_1_objects_mask = subtract_background(bkg_img, img_1)
thresh, img_2_objects_mask = subtract_background(bkg_img, img_2)
thresh, img_3_objects_mask = subtract_background(bkg_img, img_3)
thresh, img_4_objects_mask = subtract_background(bkg_img, img_4)
thresh, img_5_objects_mask = subtract_background(bkg_img, img_5)

img_1_objects_mask = get_object_mask(img_1_objects_mask)
img_2_objects_mask = get_object_mask(img_2_objects_mask)
img_3_objects_mask = get_object_mask(img_3_objects_mask)
img_4_objects_mask = get_object_mask(img_4_objects_mask)
img_5_objects_mask = get_object_mask(img_5_objects_mask)

img_1_objects = cv.bitwise_and(img_1, img_1, mask=img_1_objects_mask)
img_2_objects = cv.bitwise_and(img_2, img_2, mask=img_2_objects_mask)
img_3_objects = cv.bitwise_and(img_3, img_3, mask=img_3_objects_mask)
img_4_objects = cv.bitwise_and(img_4, img_4, mask=img_4_objects_mask)
img_5_objects = cv.bitwise_and(img_5, img_5, mask=img_5_objects_mask)

cv.imshow("img_3", img_3_objects)
cv.waitKey()

img_1_objects = cv.cvtColor(img_1_objects, cv.COLOR_BGR2GRAY)
img_2_objects = cv.cvtColor(img_2_objects, cv.COLOR_BGR2GRAY)
img_3_objects = cv.cvtColor(img_3_objects, cv.COLOR_BGR2GRAY)
img_4_objects = cv.cvtColor(img_4_objects, cv.COLOR_BGR2GRAY)
img_5_objects = cv.cvtColor(img_5_objects, cv.COLOR_BGR2GRAY)

object_images = [
    img_1_objects,
    img_2_objects,
    img_3_objects,
    img_4_objects,
    img_5_objects
]

object_keypoints = [
    scissors_keypoints,
    gum_keypoints,
    sunglasses_keypoints,
    mouse_keypoints
]

object_descriptors = [
    scissors_descriptors,
    gum_descriptors,
    sunglasses_descriptors,
    mouse_descriptors
]

object_names = [
    "scissors",
    "gum",
    "sunglasses",
    "mouse"
]

for object_image in object_images:
    for i in range(4):
        detect_object(object_keypoints[i], object_descriptors[i], object_image, all_img, True, "matches "+object_names[i])

