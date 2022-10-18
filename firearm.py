
import cv2

import urllib.request

import numpy as np


firearm_cascade = cv2.CascadeClassifier("/Users/zximy/Desktop/cascade.xml") /* Haar Cascade file */ 


url='http://192.168.0.108/640x480.jpg'

cv2.namedWindow("SCANNING AREA", cv2.WINDOW_AUTOSIZE)


while True:

    videoResponse = urllib.request.urlopen(url)

    imgnp = np.array(bytearray(videoResponse.read()), dtype=np.uint8)

    videostream = cv2.imdecode(imgnp, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gundetected = firearm_cascade.detectMultiScale(gray,scaleFactor=12, minNeighbors=26)

    for x, y, w, h in gundetected:

        videostream = cv2.rectangle(videostream, (x, y), (x+w, y+h), (0, 0, 255), 3)


    cv2.imshow("Scanning", videostream)

    Key  = cv2.waitKey(5)

    if key == ord('q'):

        break
