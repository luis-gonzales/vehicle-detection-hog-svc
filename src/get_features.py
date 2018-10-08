import cv2
import numpy as np

# Define HOG descriptor
window_size = (64,64)
block_size = (32,32)	# total size of block
block_stride = (16,16)
cell_size = (16,16)		# in pixels
n_bins = 11

hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, n_bins, 1, -1, 0, 0.2, False, 64, True)

def get_features(img):

	# Perform colorspace conversion and compute HOG features per channel
	img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

	hog0 = hog.compute(img_yuv[:,:,0]).T
	hog1 = hog.compute(img_yuv[:,:,1]).T
	hog2 = hog.compute(img_yuv[:,:,2]).T

	return np.concatenate((hog0, hog1, hog2), axis=1) # features are in [0.0, 1.0]