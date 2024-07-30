import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

#initialize the video
cap = cv2.VideoCapture('Videos/vid (4).mp4')

#the color finder
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

#variables
positionListX, positionListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False


while True:
    #getting the image
    success, img = cap.read()

    #img = cv2.imread("Ball.png")
    img = img[0:900, :]
    imgPrediction = img.copy()

    #getting the color of the ball
    imgBall, mask = myColorFinder.update(img, hsvVals)

    #getting the location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        positionListX.append(contours[0]['center'][0])
        positionListY.append(contours[0]['center'][1])

    if positionListX:
        #polynomial regression y = Ax^2 + Bx + C
        #getting the coefficients of the equation
        A, B, C = np.polyfit(positionListX, positionListY, 2)

        for i, (positionX, positionY) in enumerate(zip(positionListX, positionListY)):
            position = (positionX, positionY)
            cv2.circle(imgContours, position, 10, (0,255,0), cv2.FILLED)

            if i==0:
                cv2.line(imgContours, position, position, (0, 0, 255), 2)
            else:
                cv2.line(imgContours, (positionListX[i-1],positionListY[i-1]), position, (0, 0, 255), 2)


        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)


        if len(positionListX) < 10:
            # Prediction
            # X values 330 to 430  Y 590
            a = A
            b = B
            c = C - 590

            x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 430

        if prediction:
            cvzone.putTextRect(imgContours, "Scores", (50, 150),
                                    scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 150),
                                    scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgContours)
    cv2.waitKey(100)


