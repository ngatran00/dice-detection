#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import os
import platform
import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2

from copy import deepcopy
from sklearn.cluster import MeanShift

def find_static_frame(video_capture):
#    Find frame where all dice are rolled and stable
#    Assumming this frame is in the middle half of the video
    old_frame = None

    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cut_point = video_length/4
    cut_point_average = int(video_length/50)
    min_diff = 100000000000
    frame_id = 0

    h = None
    while(video_capture.isOpened()):
        ret, frame = video_capture.read()
        frame_no = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

        if ret is True:
            if h is None:
                h, w, c = frame.shape
                arr = np.zeros((h,w,c),np.float)
            if cut_point < frame_no < (video_length-cut_point):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if old_frame is not None:
                    dist = np.sqrt(np.sum(np.square(np.subtract(gray, old_frame))))
                    if dist < min_diff:
                        min_diff = dist
                        chosen_frame = frame
                        frame_id = frame_no
                old_frame = gray
            elif frame_no <= cut_point_average or frame_no>=video_length-cut_point_average:
                arr=arr+frame/(1+2*cut_point_average)
        else:
            break

    average_frame=np.array(np.round(arr),dtype=np.uint8)
    diff = cv2.absdiff(average_frame,chosen_frame)
    video_capture.release()
    cv2.destroyAllWindows()
    return chosen_frame, frame_id, diff

def find_if_close(cnt1,cnt2):
#    Find if 2 contours are close to each other
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 1000 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def find_contour(diff):
#    Find contour area where all dice are in
    h, w, c = diff.shape
    edges = cv2.Canny(diff,100,200)
    blur=((3,3),1)
    erode_=(5,5)
    dilate_=(7, 7)
    img = cv2.dilate(cv2.GaussianBlur(edges/255, blur[0], blur[1]), np.ones(erode_), np.ones(dilate_), iterations=2)*255
    eroded = cv2.erode(img, None, iterations=1)

    cnts = cv2.findContours(np.uint8(eroded), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    contours = []
    # loop over the contours individually
    for c in cnts:
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        w_th = w/7
        h_th = h/10

        if cv2.contourArea(c) < 300 or not(w_th<cx<w-w_th) or not (h_th<cy<h-h_th):
            continue

        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        contours.append(c)

    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    area = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
            epsilon = 0.01*cv2.arcLength(hull,True)
            approx = cv2.approxPolyDP(hull,epsilon,True)
            area.append(cv2.contourArea(approx))

    zipped = sorted(zip(area, unified), reverse=True)
    area, unified = zip(*zipped)

    return unified[0]


def detect_blob(chosen_frame):
#    Detect dots on the dice
    cX = []
    cY = []
    sizes = []

    gray = cv2.cvtColor(chosen_frame, cv2.COLOR_BGR2GRAY)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True
    params.minCircularity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    params.minThreshold = 0
    params.maxThreshold = 255

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray)
    deleted = []
    for i in range(len(keypoints)):
        if keypoints[i].size < 30:
            cX.append(keypoints[i].pt[0])
            cY.append(-1*keypoints[i].pt[1])
            sizes.append(keypoints[i].size)
        else:
            deleted.append(i)
    for index in sorted(deleted, reverse = True):
        keypoints.pop(index)

    # invert image
    deleted = []
    keypoints = detector.detect(255-gray)
    for i in range(len(keypoints)):
        if keypoints[i].size < 30:
            cX.append(keypoints[i].pt[0])
            cY.append(-1*keypoints[i].pt[1])
            sizes.append(keypoints[i].size)
        else:
            deleted.append(i)
    for index in sorted(deleted, reverse = True):
        keypoints.pop(index)

    deleted = []
    mean_keypoint_size = np.mean(np.asarray(sizes))
    for i, size in enumerate(sizes):
        if size < mean_keypoint_size -2:
            deleted.append(i)
    for index in sorted(deleted, reverse = True):
        cX.pop(index)
        cY.pop(index)
        sizes.pop(index)
    datapoints = np.array(list(zip(cX, cY)))
    return datapoints, sizes

def detect_dice(video_capture):
#    Detect dice and their value
    chosen_frame, reference_frame_no, diff = find_static_frame(video_capture)
    cnt = find_contour(diff)

    color = [255, 255, 255]
    stencil = np.zeros(chosen_frame.shape).astype(chosen_frame.dtype)
    cv2.fillPoly(stencil, [cnt], color)
    result = cv2.bitwise_and(chosen_frame, stencil)

    datapoints, sizes = detect_blob(result)

    mean_keypoint_size = np.mean(np.asarray(sizes))
    bandwidth = mean_keypoint_size*3
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(datapoints)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    detected_dice = []
    for ii, center in enumerate(cluster_centers):
        top_left_x = int(center[0]-(bandwidth))
        top_left_x = top_left_x if top_left_x >= 0 else 0
        top_left_y = -1*int(center[1]+(bandwidth))
        top_left_y = top_left_y if top_left_y >= 0 else 0
        top_left = (top_left_x, top_left_y)

        bot_right_x = int(center[0]+(bandwidth))
        bot_right_x = bot_right_x if bot_right_x >= 0 else 0
        bot_right_y = -1*int(center[1]-(bandwidth))
        bot_right_y = bot_right_y if bot_right_y >= 0 else 0
        bot_right = (bot_right_x, bot_right_y)

        count = (labels.tolist()).count(ii)
        detected_dice.append((int(center[0]),
                              int(-1*center[1]), count))
        cv2.rectangle(chosen_frame,top_left,bot_right,(0,255,0),3)
        cv2.putText(chosen_frame, str(count),
                    (int(center[0]+50),
                    int(-1*center[1]+30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (57,255,20), 2)
                    
    cv2.imshow("Result", cv2.resize(chosen_frame, (960, 540)) )
    cv2.waitKey(0)

    return reference_frame_no, detected_dice, chosen_frame

video = "2018-10-08@13-41-12.avi"

frames = []

video_capture = cv2.VideoCapture(video)
if not video_capture.isOpened():
    sys.stderr.write("Failed to open video file\"" + video_filename + "\"!\n")
    sys.exit(1)

reference_frame_no, detected_dice, chosen_frame = detect_dice(video_capture)

#plt.subplot(121),plt.imshow(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB))
#plt.title("Example of side pips considered as a new dice"), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
#plt.title("Example of side pips clustered with the pips from top face"),plt.xticks([]), plt.yticks([])
#plt.show()

cv2.destroyAllWindows()
