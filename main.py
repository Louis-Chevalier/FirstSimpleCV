import cv2
import matplotlib.pyplot as plt

#preprocess the image
image = cv2.imageread("")
plt.imshow(cv2.cvtColor(image, cv2, COLOR_BGR2RGB))
plt.show()

#Feature Extraction
resized_image = cv2.resize(image, (width, height))
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5,5),0)

#object detection
#Hough Circle Transform
circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp =1,
        minDist=50,
        param1 =50,
        param2=30,
        minRadius=10,
        maxRadius=100
)

if circles is not None:
    circles =np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(resized_image, (i[0],i[1]),i[2], (0,255,0),2)

#display results
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.show()

                

