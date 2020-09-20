import cv2
import glob
import numpy as np

image_1 = cv2.imread('/home/adrian/Desktop/Python/Personal/Deep/Final-Project/results/Test/4.jpg')
image_1 = cv2.resize(image_1, (224, 224))

image_2 = cv2.imread('/home/adrian/Desktop/Python/Personal/Deep/Final-Project/results/Test/5.jpg')
image_3 = cv2.imread('/home/adrian/Desktop/Python/Personal/Deep/Final-Project/results/Test/6.jpg')

print(image_1.shape)
print(image_2.shape)
print(image_3.shape)


white = np.zeros((224, 20, 3), dtype=np.uint8)
white.fill(255) 

final = np.hstack((image_1, white))
final = np.hstack((final, image_2))
final = np.hstack((final, white))
final = np.hstack((final, image_3))

cv2.imwrite("specific-3-4-5.jpg", final)
cv2.imshow('Main', final)
cv2.waitKey(0)