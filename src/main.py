import cv2
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from get_features import get_features
from moviepy.editor import VideoFileClip
from utils import bbox_queue, reduced_bboxes, draw_boxes_img, draw_boxes_vid

# Load classifier from pickle
infile = open("pickled_classifier",'rb')
dict_load = pickle.load(infile)
infile.close()
clf = dict_load['classifier']


def detect(img, x_start_stop, y_start_stop, window_sz, overlap):
    y_start = y_start_stop[0]
    y_stop = y_start_stop[1] - window_sz[1]
    x_start = x_start_stop[0]
    x_stop = x_start_stop[1] - window_sz[0]

    # Compute the number of pixels per step in x/y
    x_step = int(window_sz[0] * (1-overlap[0]))
    y_step = int(window_sz[1] * (1-overlap[1]))

    # Step through image at current scale and aspect ratio; append if positive prediction
    corners = []
    for y in np.arange(y_start, y_stop, y_step):
        for x in np.arange(x_start, x_stop, x_step):
            cur_img = img[y:y+window_sz[1], x:x+window_sz[0]]
            cur_img_resize = cv2.resize(cur_img, (64,64))

            if clf.predict(get_features(cur_img_resize)):
                corners.append( np.array([x, y, x+window_sz[0], y+window_sz[1]]) )
            #corners.append( np.array([x, y, x+window_sz[0], y+window_sz[1]]) )

    return corners


def vehicle_detect(img):

    # Step through image at multiple scales
    corners = []
    corners.extend(detect(img, x_start_stop=(260,1030), y_start_stop=(400,490), window_sz=(80,64), overlap=(0.6,0.75)))
    corners.extend(detect(img, x_start_stop=(20,1300), y_start_stop=(400,550), window_sz=(120,96), overlap=(0.8,0.66)))
    corners.extend(detect(img, x_start_stop=(5,1300), y_start_stop=(400,580), window_sz=(160,128), overlap=(0.6,0.75)))
    corners.extend(detect(img, x_start_stop=(10,1300), y_start_stop=(400,680), window_sz=(240,192), overlap=(0.6,0.65)))

    # Draw boxes according to whether image or video
    if input_is_jpg: return draw_boxes_img(img, corners)
    return draw_boxes_vid(img, corners)


if __name__ == "__main__":

    # Parse command line input
    img_path = sys.argv[1]          # e.g., input/image.jpg
    f_name = img_path.split('/')[1] # e.g., image.jpg
    ext = img_path.split('.')[1]    # e.g., jpg
  
    # Process according to `jpg` or `mpg`
    if ext == 'jpg':
        input_is_jpg = True
        img = cv2.imread(img_path)[:,:,::-1] # RGB
        lane_detect = vehicle_detect(img)    
        plt.imsave('output/' + f_name, lane_detect, vmin=0, vmax=255)

    elif ext == 'mp4':
        input_is_jpg = False
        vid_clip = VideoFileClip(img_path)
        vid_result = vid_clip.fl_image(vehicle_detect)
        vid_result.write_videofile('output/' + f_name, audio=False, progress_bar=True)

    else:
        print("invalid input format")