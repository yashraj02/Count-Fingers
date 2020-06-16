import cv2 as cv
import numpy as np
from sklearn.metrics import pairwise

background = None
accumulated_weight = 0.5
roi_top = 20
roi_bot = 300
roi_right = 300
roi_left = 600


def cal_acc_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    cv.accumulateWeighted(frame, background, accumulated_weight)


# Segmenting Hand in ROI
def segment(frame, threshold_min=25):
    diff = cv.absdiff(background.astype('uint8'), frame)
    ret, threshold = cv.threshold(diff, threshold_min, 255, cv.THRESH_BINARY)

    image, contours, hierarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # finding the largest contour
    hand_segment = max(contours, key=cv.contourArea)
    return (threshold, hand_segment)


def count_fingers(thresholded, hand_segment):
    # Calculated the convex hull of the hand segment
    conv_hull = cv.convexHull(hand_segment)

    # Now the convex hull will have at least 4 most outward points, on the top, bottom, left, and right.
    # Let's grab those points by using argmin and argmax. Keep in mind, this would require reading the documentation
    # And understanding the general array shape returned by the conv hull.

    # Find the top, bottom, left , and right.
    # Then make sure they are in tuple format
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull

    # Calculate the Euclidean Distance between the center of the hand and the left, right, top, and bottom.
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]

    # Grab the largest distance
    max_distance = distance.max()

    # Create a circle with 90% radius of the max euclidean distance
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # Not grab an ROI of only that circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    # draw the circular ROI
    cv.circle(circular_roi, (cX, cY), radius, 255, 10)

    # Using bit-wise AND with the cirle ROI as a mask.
    # This then returns the cut out obtained using the mask on the thresholded hand image.
    circular_roi = cv.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Grab contours in circle ROI
    image, contours, hierarchy = cv.findContours(circular_roi.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Finger count starts at 0
    count = 0

    # loop through the contours to see if we count any more fingers.
    for cnt in contours:

        # Bounding box of countour
        (x, y, w, h) = cv.boundingRect(cnt)

        # Increment count of fingers based on two conditions:

        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))

        # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
        limit_points = ((circumference * 0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1

    return count


cap = cv.VideoCapture(0)
num_frames = 0
while True:
    ret, frame = cap.read()
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bot, roi_right:roi_left]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    if num_frames < 60:
        cal_acc_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv.putText(frame_copy, 'Wait. Getting Background', (200, 300), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
        hand = segment(gray)
        if hand is not None:
            threshold, hand_segment = hand
            cv.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 5)
            fingers = count_fingers(threshold, hand_segment)
            cv.putText(frame_copy, str(fingers), (70, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv.imshow('Threshold', threshold)
    cv.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bot), (255, 0, 0), 5)
    num_frames += 1
    cv.imshow('Finger Count', frame_copy)
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()
