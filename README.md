# Driver Drowsiness Detection System

## Introduction

This Python script uses OpenCV along with the cvzone library to create a real-time driver drowsiness detection system. It detects facial landmarks using the FaceMesh module and plots the ratio between specific landmarks to determine if the driver is potentially drowsy.

## Optimization for Sleep Detection

The code is optimized to prioritize the detection of driver sleepiness over other potential eye movements like looking left or right. Here’s how the optimization is achieved:

1. **Facial Landmarks and Ratios**: The code utilizes facial landmarks detected by the FaceMesh module to calculate ratios between specific landmarks. These ratios are used to determine if the driver’s eyes are closed for an extended period, indicating drowsiness.

2. **Thresholding and Detection Logic**: Threshold values are employed to distinguish between normal eye movements (such as looking left or right) and prolonged closure of the eyes indicative of drowsiness. By setting appropriate threshold values for the eye closure ratios, the code can better differentiate between different states of the driver.

3. **Continuous Monitoring and Alerting**: The script continuously monitors the calculated ratios and maintains a record of consecutive frames indicating drowsiness. This approach ensures that the system alerts the driver only when there is consistent evidence of sleepiness, reducing false positives that might occur due to brief eye closures during normal activities.

4. **Adaptability and Fine-tuning**: The code allows for adjustments and fine-tuning of parameters such as the threshold values for eye closure ratios. This flexibility enables optimization for different individuals and environmental conditions, ensuring reliable detection of drowsiness while minimizing false alarms.

## Installing prerequisites

Before running the script, make sure you have installed the required dependencies:

```bash
pip install cvzone
pip install mediapipe --user
```
## The entire code
```bash
import cv2 as cv
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
dangerDetector = []
driverSleeping = False

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        for id in idList:
            cv.circle(img, face[id], 4, (255, 0, 255), cv.FILLED)
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthVertical, _ = detector.findDistance(leftUp, leftDown)
        lengthHorizontal, _ = detector.findDistance(leftLeft, leftRight)
        cv.line(img, leftLeft, leftRight, (0, 200, 0), 2)
        cv.line(img, leftUp, leftDown, (0, 200, 0), 2)
        ratio = int(100 * (lengthVertical / lengthHorizontal))
        if ratio <= 29:
            success, img = cap.read()
            img = cv.flip(img, 1)
            img, faces = detector.findFaceMesh(img, draw=False)
            if faces:
                face = faces[0]
                for id in idList:
                    cv.circle(img, face[id], 4, (255, 0, 255), cv.FILLED)
                leftUp = face[159]
                leftDown = face[23]
                leftLeft = face[130]
                leftRight = face[243]
                lengthVertical, _ = detector.findDistance(leftUp, leftDown)
                lengthHorizontal, _ = detector.findDistance(leftLeft, leftRight)
                cv.line(img, leftLeft, leftRight, (0, 200, 0), 2)
                cv.line(img, leftUp, leftDown, (0, 200, 0), 2)
                ratio = int(100 * (lengthVertical / lengthHorizontal))
                if ratio >= 31:
                    dangerDetector.clear()
                else:
                    dangerDetector.append(1)
        if len(dangerDetector) >= 15:
            print("Driver is Sleeping!")
            driverSleeping = True          
        else:
            if driverSleeping:
                print("Driver is Awake!")
                driverSleeping = False
        imgPlot = plotY.update(ratio)
        imgPlot = cv.resize(imgPlot, (640, 480))
    imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    cv.imshow("Image", imgStack)
    key = cv.waitKey(1)
    if key == ord('x'):
        break
cap.release()
cv.destroyAllWindows()
```

## Clone the repository
```bash
git clone https://github.com/Sayan-03/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```
