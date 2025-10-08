import cv2
import numpy as np

img = cv2.imread("/Applications/Work Space/Python Work Space/python_computer_vision/ytü.png")

img_hor = np.hstack((img, img))
img_var = np.vstack((img, img))


cv2.imshow("Horizontal", img_hor)
cv2.imshow("Vertical", img_var)
cv2.waitKey(0)