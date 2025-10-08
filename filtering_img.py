import matplotlib.pyplot as plt
import matplotlib.image as mpimg # image related ops
import numpy as np
import cv2 # opencv lib

img_path = "/Applications/Work Space/Python Work Space/python_computer_vision/ytü.png"

# car1 = mpimg.imread(img_path) # Read RGB 
# print(car1.shape)
# plt.imshow(car1)
# plt.show()

car1_cv2 = cv2.imread(img_path) # Read BGR and we should convert to RGB
# print(car1_cv2.shape)
# plt.imshow(car1_cv2)
# plt.show()

# car1_cv2_BGR2RGB = cv2.cvtColor(car1_cv2, cv2.COLOR_BGR2RGB)
# plt.imshow(car1_cv2_BGR2RGB)
# plt.show()

# car1_cv2_BGR2GRAY = cv2.cvtColor(car1_cv2, cv2.COLOR_BGR2GRAY)
# plt.imshow(car1_cv2_BGR2GRAY, cmap="gray")
# plt.show()   


## understanding composition of colored images
def viusalize_RGB_channel(imgArray=None, figsize=(10,7)):
  # splitting the RGB components
  B, G, R = cv2.split(imgArray)

  # create zero matrix of shape of image
  Z = np.zeros(B.shape, dtype=B.dtype) # can use any channel

  # init subplots
  fig, ax = plt.subplots(2,2, figsize=figsize)

  # plotting the actual image and RGB images
  [axi.set_axis_off() for axi in ax.ravel()]

  ax[0,0].set_title("Original Image")
  # ax[0,0].set_axis_off()
  ax[0,0].imshow(cv2.merge((R,G,B)))

  ax[0,1].set_title("Red Ch Image")
  ax[0,1].imshow(cv2.merge((R,Z,Z)))

  ax[1,0].set_title("Green Ch Image")
  ax[1,0].imshow(cv2.merge((Z,G,Z)))

  ax[1,1].set_title("Blue Ch Image")
  ax[1,1].imshow(cv2.merge((Z,Z,B)))

  plt.show()


# viusalize_RGB_channel(imgArray = car1_cv2)



# # Random imaged
random_colored_img = np.random.randint(0, 255, (6,6,3))
random_colored_img.shape
# plt.imshow(random_colored_img)
# plt.show()

# viusalize_RGB_channel(imgArray = random_colored_img)




# # Understanding the convolution

sobel = np.array([[ 1, 0,-1],
                  [ 2, 0,-2],
                  [ 1, 0,-1]])

print("highlighting Vertical edges:\n", sobel)

print("highlighting Horizontal edges:\n", sobel.T)

example1 = [
    [0,0,0,0,255,255,255,255,0,0,0,0],
    [0,0,0,0,255,255,255,255,0,0,0,0],
    [0,0,0,0,255,255,255,255,0,0,0,0],
    [0,0,0,0,255,255,255,255,0,0,0,0],
    [0,0,0,0,255,255,255,255,255,255,255,255],
    [0,0,0,0,255,255,255,255,255,255,255,255],
    [0,0,0,0,255,255,255,255,255,255,255,255],
    [0,0,0,0,255,255,255,255,255,255,255,255],
    [0,0,0,0,255,255,255,255,0,0,0,0],
    [0,0,0,0,255,255,255,255,0,0,0,0],
    [0,0,0,0,255,255,255,255,0,0,0,0],
    [0,0,0,0,255,255,255,255,0,0,0,0],
            ]

example1 = np.array(example1)

plt.imshow(example1, cmap="gray")
plt.show()




def simple_conv(imgFilter=None, picture=None):
  # extract the shape of the image
  p_row, p_col = picture.shape

  k = imgFilter.shape[0] # k =3

  temp = list()

  stride = 1

  # resulant image size
  final_cols = (p_col - k)//stride + 1
  final_rows = (p_row - k)//stride + 1

  # take vertically down stride across row by row
  for v_stride in range(final_rows):
    # take horizontal right stride across col by col
    for h_stride in range(final_cols):
      target_area_of_pic = picture[v_stride: v_stride + k, h_stride: h_stride + k]
      z = sum(sum(imgFilter * target_area_of_pic))
      temp.append(z)

  resulant_image = np.array(temp).reshape(final_rows, final_cols)
  return resulant_image



# Horizontal 
results = simple_conv(imgFilter=sobel, picture=example1)

plt.imshow(results, cmap="gray")
plt.title("Hozirzontal Convolution")
plt.show()


# Vertical 
results = simple_conv(imgFilter=sobel.T, picture=example1)

plt.imshow(results, cmap="gray")
plt.title("Vertical Convolution")
plt.show()



# Example of the Horizontal and vectoral convolution of the images
car1_cv2_BGR_GRAY = cv2.cvtColor(car1_cv2, cv2.COLOR_BGR2GRAY)
plt.imshow(car1_cv2_BGR_GRAY, cmap="gray")
plt.title("Original Images")
plt.show()

result = simple_conv(imgFilter=sobel, picture=car1_cv2_BGR_GRAY)
plt.imshow(result, cmap="gray")
plt.title("Hozirzontal Convolution")
plt.show()


result = simple_conv(imgFilter=sobel.T, picture=car1_cv2_BGR_GRAY)
plt.imshow(result, cmap="gray")
plt.title("Vertical Convolution")
plt.show()
