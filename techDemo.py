import cv2
import numpy as np
from cmu_112_graphics import *
def appStarted(app):
    app.defectsRadius = 10
    app.centerRadius = 2
    app.thickness = 5

def drawCircles(app, canvas):
    

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    _, frame = cap.read()
    #face subtraction
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x,y,w,h) in faces:
    #     faceMask = np.zeros((h, w, 3))
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_color = frame[y:y+h, x:x+w]
    #     frame[y:y+h, x:x+w] = 0

    #background subtraction
    fgmask = fgbg.apply(frame)
    forground = cv2.bitwise_and(frame, frame, mask=fgmask)
    #cv2.imshow("Forground", forground)
    hsv_frame = cv2.cvtColor(forground, cv2.COLOR_BGR2HSV)
        # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
        # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
        # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

        # Skin Color
    low = np.array([0, 24, 145])
    high = np.array([179, 114, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #cv2.imshow("SkinColor Mask", result)

    #imfill
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_floodfill_inv = cv2.GaussianBlur(im_floodfill_inv, (13,13), 7)
    #cv2.imshow("flood filled result", im_floodfill_inv)

    #cleaning up 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    foreground = cv2.morphologyEx(im_floodfill_inv, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Cleaned up foreground", foreground)

    #find contours
    CONTOURS, _ = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    newContours = []
    for c in CONTOURS:
        area = cv2.contourArea(c)
        #print(area)
        if area > 28000:
            newContours.append(c)
    cv2.drawContours(frame, newContours, -1, (0,255,0), 3)
    #cv2.imshow('hand contour', frame)

    #Convex Hull
    hull = [cv2.convexHull(c) for c in newContours] 
    cv2.drawContours(frame, hull, -1, (0,0,255), 3)

    cv2.imshow('Convex Hull', frame)

    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(newContours)
    boundRect = [None]*len(newContours)
    centers = [None]*len(newContours)
    radius = [None]*len(newContours)
    for i, c in enumerate(newContours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    try:
        center = centers[0]
        r = radius[0]
        r = int(r)
        x = int(center[0])
        y = int(center[1])
        print(f'r now is {int(r)} and center now is {int(center[0])} {int(center[1])}')
    except:
        pass
    # Identify the palm center
    if center != None:
        cv2.circle(frame, (x,y), 2, (255,0,0), 5)
    cv2.imshow('frame', frame)

    # convex hull defects
    try:
        max_cont = max(newContours, key=cv2.contourArea)
    except:
        pass
    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        #cv2.circle(frame, cnt_centroid, 2, (0,255,0), 5)
        #print(f'defects is of type {type(defects)}, {defects}, the shape of the defects is {np.shape(defects)}')

    #plotting convex hull defects
    for i in range(len(defects)):
        s,e,f,d = defects[i][0]
        x, y = max_cont[f][0]
        #print(type(max_cont[f]))
        #print(np.shape(max_cont[f]))
        #print(f'x and y is ({x},{y})')
        cv2.circle(frame, (x,y), 10, (255,0,0), 5)
    cv2. imshow('frame-convex hull defects pts', frame)
           
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





