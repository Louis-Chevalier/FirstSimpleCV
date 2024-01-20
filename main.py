import cv2
import numpy as np
import matplotlib.pyplot as plt

#preprocess the image
image = cv2.imread("resources/LandingPad.jpg")
if image is None:
    print("Error: Image not loade")
else:

    #Feature Extraction
    width =800
    height =800
    resized_image = cv2.resize(image, (width, height))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5,5),0)

    #object detection
    #Hough Circle Transform
    circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp =1,          #Inverse ration of the accumulator resolution to the
                            #img resolution (1 means same as input)
            minDist=50,     #Minimum distance between detected centers
            param1 =80,     #Upperthreshold for the internal Canny edge
                            #detector
            param2=90,      #Threshold for center detection. Lower value means
                            #more circles will be detected (increase for more
                            #circles
            minRadius=200,   #Minimum radius of detected circle
            maxRadius=400   #Max radius of the detected circle
    )

    if circles is not None:
        circles =np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(resized_image, (i[0],i[1]),i[2], (0,255,0),2)

    #display results
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()

                

