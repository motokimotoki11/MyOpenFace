#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2020 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Tested on Ubuntu 18.04.3 LTS, OpenCV 4.1.2, Python 3
#Comparison of 4 different methods on corner detection.
#Parameters set such that all the methods will find around 500 keypoints in the video.
#You need a video named "video.mp4" in your script folder for running the code.
#Videos sourced from: https://www.pexels.com/videos

import cv2
import numpy as np
from operator import itemgetter 

#print(cv2.__version__)
video_capture = cv2.VideoCapture("./video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter("./original.avi", fourcc, 24.0, (3840,2160))
out_harris = cv2.VideoWriter("./harris.avi", fourcc, 24.0, (3840,2160))
out_shitomasi = cv2.VideoWriter("./shitomasi.avi", fourcc, 24.0, (3840,2160))
out_fast = cv2.VideoWriter("./fast.avi", fourcc, 24.0, (3840,2160))
out_orb = cv2.VideoWriter("./orb.avi", fourcc, 24.0, (3840,2160))

while(True):
    ret, frame = video_capture.read()
    if(frame is None): break #check for empty frames (en of video)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ##Harris
    mask_harris = cv2.cornerHarris(np.float32(frame_gray), blockSize=2, ksize=3, k=0.04) #2, 3, 0.04 // 2, 5, 0.07
    #mask_harris = cv2.dilate(mask_harris, None)    
    cutout = np.sort(mask_harris.flatten())[-500] #sort from smaller to higher, then take index for cutout
    corners = np.where(mask_harris > cutout)
    corners = zip(corners[0], corners[1])
    kp = list()
    for i in corners: kp.append(cv2.KeyPoint(i[1], i[0], 20))
    frame_harris = cv2.drawKeypoints(frame_gray, kp, None, [0, 0, 255], 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            
    ##Shi-Tomasi
    #maxCorners: Maximum number of corners to return
    #qualityLevel: Parameter characterizing the minimal accepted quality of image corners.
    #minDistance: Minimum possible Euclidean distance between the returned corners.
    #blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.

    corners = cv2.goodFeaturesToTrack(frame_gray, maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=2)
    corners = np.int0(corners)
    kp = list()
    for i in corners: kp.append(cv2.KeyPoint(i.ravel()[0], i.ravel()[1], 20))
    frame_shitomasi = cv2.drawKeypoints(frame_gray, kp, None, [0, 0, 255], 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    ##FAST
    #It is several times faster than other existing corner detectors. 
    #But it is not robust to high levels of noise. It is dependent on a threshold.
    #threshold: a threshold over the point to keep
    #nonmaxSuppression: wheter non-maximum suppression is to be applied or not
    #Neighborhood (three flags) cv.FAST_FEATURE_DETECTOR_TYPE_5_8 / 7_12 / 9_16
    #frame_fast = np.copy(frame)
    #Here I choose the magic number 135 for the threshold so that it finds around 500 corners.
    fast = cv2.FastFeatureDetector_create(threshold=165, nonmaxSuppression=True, 
                                          type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16) #default is TYPE_9_16
    kp = fast.detect(frame_gray, None)
    #print(len(kp)) #use this print to check how many keypoints are found by FAST
    ##Uncomment these two lines if you want to randomly pick 500 keypoints
    #indices = np.random.choice(len(kp), 500, replace=False)
    #kp = itemgetter(*indices.tolist())(kp)
    for i in kp: i.size=20 #changing the diameter to make it coherent with the other methods
    frame_fast = cv2.drawKeypoints(frame_gray, kp, None, [0, 0, 255], 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ##ORB (Oriented FAST and Rotated BRIEF)
    #First it use FAST to find keypoints, then apply Harris corner measure 
    #to find top N points among them.
    #nFeatures: maximum number of features to be retained
    #scoreType: whether Harris score or FAST score to rank the features (default: Harris)
    orb = cv2.ORB_create(nfeatures=500)
    kp = orb.detect(frame_gray, None)
    kp, des = orb.compute(frame_gray, kp)
    frame_orb = cv2.drawKeypoints(frame_gray, kp, None, [0, 0, 255], 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #Writing in the output file
    #out.write(frame)
    out_harris.write(frame_harris)
    out_shitomasi.write(frame_shitomasi)
    out_fast.write(frame_fast)
    out_orb.write(frame_orb)
        
    #Showing the frame and waiting for the exit command
    #cv2.imshow('Original', frame) #show on window
    cv2.imshow('Harris', frame_harris) #show on window
    cv2.imshow('Shi-Tomasi', frame_shitomasi) #show on window
    cv2.imshow('FAST', frame_fast) #show on window
    cv2.imshow('ORB', frame_orb) #show on window
    if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed


#Release the camera
video_capture.release()
print("Bye...")    
