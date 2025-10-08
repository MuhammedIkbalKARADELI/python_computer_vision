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


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


import numpy as np

KERNELS = {
    # --- Kenar (gradyan) operatörleri ---
    "sobel_x": np.array([[ -1, 0, 1],
                         [ -2, 0, 2],
                         [ -1, 0, 1]], dtype=float),

    "sobel_y": np.array([[ -1,-2,-1],
                         [  0, 0, 0],
                         [  1, 2, 1]], dtype=float),

    "scharr_x": np.array([[  -3, 0,  3],
                          [ -10, 0, 10],
                          [  -3, 0,  3]], dtype=float),

    "scharr_y": np.array([[  -3, -10, -3],
                          [   0,   0,  0],
                          [   3,  10,  3]], dtype=float),

    "prewitt_x": np.array([[ -1, 0, 1],
                           [ -1, 0, 1],
                           [ -1, 0, 1]], dtype=float),

    "prewitt_y": np.array([[ -1,-1,-1],
                           [  0, 0, 0],
                           [  1, 1, 1]], dtype=float),

    "roberts_x": np.array([[1, 0],
                           [0,-1]], dtype=float),

    "roberts_y": np.array([[0, 1],
                           [-1,0]], dtype=float),

    # --- İkinci türev / Laplace ---
    "laplacian_3x3": np.array([[ 0, -1,  0],
                               [-1,  4, -1],
                               [ 0, -1,  0]], dtype=float),

    "laplacian_3x3_cross": np.array([[-1, -1, -1],
                                     [-1,  8, -1],
                                     [-1, -1, -1]], dtype=float),

    # --- Bulanıklaştırma ---
    "box_blur_3x3": (1/9.0) * np.ones((3,3), dtype=float),

    # Gaussian yaklaşıkları (doğru Gauss için aşağıdaki 5x5'i kullan)
    "gaussian_3x3_sigma1": (1/16.0) * np.array([[1, 2, 1],
                                                [2, 4, 2],
                                                [1, 2, 1]], dtype=float),

    "gaussian_5x5_sigma1": (1/273.0) * np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ], dtype=float),

    # --- Keskinleştirme / Geliştirme ---
    "sharpen_basic": np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]], dtype=float),

    "edge_enhance": np.array([[ 0,  0,  0],
                              [-1,  1,  0],
                              [ 0,  0,  0]], dtype=float),

    "outline": np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]], dtype=float),

    "emboss": np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]], dtype=float),

    # --- Hareket bulanıklığı (yatay) ---
    "motion_blur_1x5": (1/5.0) * np.array([[1, 1, 1, 1, 1]], dtype=float),

    # --- Ortalama çıkarma (high-pass) örneği ---
    "highpass_3x3": np.array([[-1,-1,-1],
                              [-1, 8,-1],
                              [-1,-1,-1]], dtype=float),
}



def conv2d(
    picture: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: str = "valid",   # "valid" or "same"
    flip_kernel: bool = True, # True => real convolution; False => cross-correlation
    dtype: np.dtype | None = None,
):
    """
    2D convolution/cross-correlation on single-channel images.

    picture: HxW array
    kernel: khxkw array
    stride: int >= 1
    padding: "valid" or "same"
    flip_kernel: flip kernel for true convolution
    dtype: output dtype (defaults to float32 if input is integer)
    """
    assert picture.ndim == 2 and kernel.ndim == 2, "Use 2D arrays (single-channel)."

    H, W = picture.shape
    kh, kw = kernel.shape
    if flip_kernel:
        kernel = np.flip(kernel, (0, 1))

    # choose output dtype sensibly
    if dtype is None:
        if np.issubdtype(picture.dtype, np.integer):
            out_dtype = np.float32
        else:
            out_dtype = picture.dtype
    else:
        out_dtype = dtype

    # padding
    if padding == "same":
        # ‘same’ defined like many libraries: ceil(H/stride) x ceil(W/stride)
        out_H = int(np.ceil(H / stride))
        out_W = int(np.ceil(W / stride))
        pad_H = max((out_H - 1) * stride + kh - H, 0)
        pad_W = max((out_W - 1) * stride + kw - W, 0)
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pic = np.pad(picture, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    elif padding == "valid":
        pic = picture
        out_H = (H - kh) // stride + 1
        out_W = (W - kw) // stride + 1
    else:
        raise ValueError("padding must be 'valid' or 'same'")

    # sliding windows (vectorized)
    windows = sliding_window_view(pic, (kh, kw))[::stride, ::stride]  # shape: (out_H, out_W, kh, kw)

    # tensor dot product over last two dims
    out = np.tensordot(windows, kernel, axes=((2, 3), (0, 1))).astype(out_dtype)  # (out_H, out_W)
    return out


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

# result = conv2d(picture=car1_cv2_BGR_GRAY, kernel=sobel)
# plt.imshow(result, cmap="gray")
# plt.title("Hozirzontal Convolution Pro")
# plt.show()

result = simple_conv(imgFilter=sobel, picture=car1_cv2_BGR_GRAY)
plt.imshow(result, cmap="gray")
plt.title("Hozirzontal Convolution")
plt.show()

result = simple_conv(imgFilter=sobel.T, picture=car1_cv2_BGR_GRAY)
plt.imshow(result, cmap="gray")
plt.title("Vertical Convolution")
plt.show()




# Örnek kullanım (cv2 ile cross-correlation):
import cv2
out = cv2.filter2D(src=car1_cv2_BGR_GRAY, ddepth=cv2.CV_32F, kernel=KERNELS["sobel_x"])
magnitude = np.hypot(
    cv2.filter2D(car1_cv2_BGR_GRAY, cv2.CV_32F, KERNELS["sobel_x"]),
    cv2.filter2D(car1_cv2_BGR_GRAY, cv2.CV_32F, KERNELS["sobel_y"])
)
plt.imshow(out)
plt.show()
