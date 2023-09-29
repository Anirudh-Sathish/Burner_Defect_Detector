""" Script to perform watershed segmentation on the burner"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_image(img):
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def main():
    img = cv2.imread("new_image2.jpg")
    # display_image(img)
    #image grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN,kernel,iterations=2)
    display_image(binary_img)


    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # sure background area
    sure_bg = cv2.dilate(binary_img, kernel, iterations=3)
    display_image(sure_bg)
    axes[0, 0].set_title('Sure Background')
    
    # Distance transform
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    display_image(dist, axes[0,1])
    axes[0, 1].set_title('Distance Transform')
    
    #foreground area
    _ , sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)  
    display_image(sure_fg, axes[1,0])
    axes[1, 0].set_title('Sure Foreground')
    
    # unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)
    imshow(unknown, axes[1,1])
    axes[1, 1].set_title('Unknown')
  
    plt.show()

if __name__ == '__main__':
    main()