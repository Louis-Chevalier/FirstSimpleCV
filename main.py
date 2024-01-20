import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def detect_circles(params, image):
    #print("Received parameters:", params)
    #print("Number of received parameters:", len(params))

    if len(params) != 6:
        raise ValueError("Expected 6 parameters, but received {}.".format(len(params)))

    dp, minDist, param1, param2, minRadius, maxRadius =map(int, list(params))
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

def objective(params, image):
    # This function calculates the objective to be minimized (negative score)
    result = detect_circles(params, image)
    score = calculate_score(result)  # Define your scoring function

    return -score  # We use negative score since Bayesian optimization minimizes

def calculate_score(result):
    # Define a scoring function based on your specific needs
    # For example, you can count the number of detected circles or use other metrics
    return len(result) if result is not None else 0

# Load the image
image = cv2.imread("resources/LandingPad.jpg")
if image is None:
    print("Error: Image not loaded.")
else:
    # Define search space for hyperparameters
    bounds = [(1,5),(10,200),(10,100),(10,100),(5,50),(50,200)]

    #Perform Bayesian Parameters
    result = minimize(objective, [2,100,50,30,10,100], args=(image,), bounds=bounds, method ='L-BFGS-B')


    # Get the best hyperparameters
    #print("Optimization Results:")
    #print(result)
    #print("Best Hyperparameters:", result.x)
    #print("Best Objective Value (negative score):", -result.fun)


    # Display the best result using the best hyperparameters
    best_result = detect_circles(result.x, image)
    plt.imshow(cv2.cvtColor(best_result, cv2.COLOR_BGR2RGB))
    plt.show()
               
