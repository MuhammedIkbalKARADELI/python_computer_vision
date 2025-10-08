import pandas as pd
import numpy as np
import os
import PIL
import cv2


# # Reading Images
# img = cv2.imread("/Applications/Work Space/Python Work Space/python_computer_vision/ytü.png")
# print(img.shape)
# cv2.imshow("Output", img)
# cv2.waitKey(0)


# # Reading Videos
# cap = cv2.VideoCapture("/Applications/Work Space/Python Work Space/python_computer_vision/VIDEO-2025-09-23-13-27-34.mp4")

# while True:
#     success, img = cap.read()
#     print(img.shape)

# while True:
#     success, img = cap.read()
#     if not success:
#         print("Video bitti veya okunamadı.")
#         break
#     print(img.shape)
# cap.release()
# cv2.destroyAllWindows()

# while True:
#     success, img = cap.read()
#     print(img.shape)
#     cv2.imshow("Output", img)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#     if not success:
#         print("Video bitti veya okunamadı.")
#         break
#     print(img.shape)
# cap.release()
# cv2.destroyAllWindows()



# # Reading Webcam

cap = cv2.VideoCapture(0)

cap.set(3, 832) # set width
cap.set(3, 480) # set height

while True:
    success, img = cap.read()
    # aslında burda her bir image e bir manipulatiion yapalabilir. 
    print(img.shape)
    cv2.imshow("Output", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


