import cv2
import numpy as np
img = cv2.imread('D:\XuLyAnh_PTIT\imgs\dog.jpg', cv2.IMREAD_GRAYSCALE)
#Thresholding
def Thresholding(img):
    tmp_img = img
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            tmp_img[r][c] = 225 if tmp_img[r][c] >= 128 else 0
    return tmp_img

#negative_image
def negative_image(img):
    tmp_img = img
    max_value = np.max(img)
    return max_value - tmp_img

#sigmoid tranform
def sigmoid_tranform(img):
    tmp_img = img
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            tmp_img[r][c] = 225 / (1 - np.exp(tmp_img[r][c]))
    return tmp_img

cv2.imshow('tranformed_img',negative_image(img))
cv2.waitKey(0)
cv2.destroyAllWindows()