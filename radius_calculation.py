"""Script to calculate inner and outer diameter"""

# neccessary imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# implement config later
class ImageReader:
    def __init__(self,image_path):
        self.path = image_path
        self.image = self.read_image()
    def read_image(self):
        return cv2.imread(self.path)
    def convert_to_RGB(self,img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_rgb
    def convert_to_grayscale(self,img):
        image_gray = cv2.cvtColor(self.convert_to_RGB(img), cv2.COLOR_BGR2GRAY)
        return image_gray
    def convert_to_binary(self, threshold=128):
        _, image_binary = cv2.threshold(self.convert_to_grayscale(self.image), threshold, 255, cv2.THRESH_BINARY)
        return image_binary

class ImageProcessor:
    def __init__(self,file_path,threshold,image_shape):
        self.path = file_path
        self.img_reader = ImageReader(file_path)
        self.threshold = threshold
        self.image_shape = image_shape
        self.inner_circle_param1 ,self.inner_circle_param2 = 10 , 50
        self.outer_circle_param1 ,self.outer_circle_param2 = 116 , 450
    def preprocess_image(self):
        binary_image = self.img_reader.convert_to_binary(self.threshold)
        sharpened_image = cv2.Laplacian(binary_image, cv2.CV_64F)
        sharpened_image = cv2.convertScaleAbs(sharpened_image)
        final_image = cv2.resize(binary_image, self.image_shape)
        return final_image
    def __find_circle_hough__(self,img,mode):
        """
        Finds the outer radius using Hough transform.

        :param img: Image of an eye
        :type img: nd.array
        :return: x, y coordinates of the centre of the pupil and its radius
        :rtype:
        """
        if mode == 1:
            min_radius = self.inner_circle_param1 
            max_radius = self.inner_circle_param2
        else:
            min_radius = self.outer_circle_param1
            max_radius = self.outer_circle_param2
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
        if circles is None:
            print("No circles detected in the image")
            print(f"{self.path}")
            return None , None , None
        circles = np.uint16(np.around(circles))
        return circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]
    def find_hough_circle(self,mode):
        image = self.preprocess_image()
        x_coordinate , y_coordinate , radius = self.__find_circle_hough__(image,mode)
        return x_coordinate , y_coordinate , radius
    
def main():
    folder_path = "../Burner_Data"
    threshold = 70
    image_shape = (256,256)
    files = os.listdir(folder_path)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        x_coord_inner , y_coord_inner , radius_inner = ImageProcessor(
            file_path,threshold,image_shape).find_hough_circle(mode = 1)
        print(x_coord_inner,y_coord_inner,radius_inner)
        x_coord_outer , y_coord_outer , radius_outer = ImageProcessor(
            file_path,threshold,image_shape).find_hough_circle(mode = 2)
        print(x_coord_outer,y_coord_outer,radius_outer)

        



if __name__ == '__main__':
    main()
