import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius):
    # Feature Extraction
    width, height = 800, 600
    resized_image = cv2.resize(image, (width, height))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Object Detection - Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(resized_image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    return resized_image

def grid_search(image):
    # Define parameter ranges to search
    dp_values = [1]
    minDist_values = [50, 60, 70,75, 100, 110, 115]
    param1_values = [30, 40, 50, 60, 70, 80, 90]
    param2_values = [20, 30, 40, 50, 60, 70, 80, 90]
    minRadius = 10
    maxRadius = 400

    # Perform grid search
    best_result = None
    best_score = 0

    for dp in dp_values:
        for minDist in minDist_values:
            for param1 in param1_values:
                for param2 in param2_values:
                    result = detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius)
                    score = calculate_score(result)  # Define your scoring function

                    if score > best_score:
                        best_score = score
                        best_result = result

    return best_result

def calculate_score(result):
    # Define a scoring function based on your specific needs
    # For example, you can count the number of detected circles or use other metrics
    return len(result) if result is not None else 0

# Load the image
image = cv2.imread("resources/LandingPad.jpg")
if image is None:
    print("Error: Image not loaded.")
else:
    # Perform grid search and display the best result
    best_result = grid_search(image)

    # Display the best result
    plt.imshow(cv2.cvtColor(best_result, cv2.COLOR_BGR2RGB))
    plt.show()
                

