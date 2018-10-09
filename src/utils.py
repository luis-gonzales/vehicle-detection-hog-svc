import cv2
import numpy as np
from collections import deque
from scipy.ndimage.measurements import label


bbox_queue = deque([], 12)


def reduced_bboxes(bboxes, img_shape, threshold=1):
    '''
    Assumes bboxes format of [x1, y1, x2, y2]
    '''

    # Initialize heatmap
    heatmap = np.zeros(img_shape[0:2], dtype=np.uint8)
    
    # Add count for each bbox
    for box in bboxes:
        heatmap[box[1]:box[3], box[0]:box[2]] += 1

    # Perform thresholding
    heatmap[heatmap <= threshold] = 0

    # Get map of unique regions    
    act_map, num_labels = label(heatmap)

    reduced_bboxes = []
    # Iterate through all detected regions and save top-left and bottom-right coordinates
    for car_number in range(1, num_labels+1):
        nonzero = (act_map == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        reduced_bboxes.append([np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)])

    #return bboxes
    return reduced_bboxes


def draw_boxes_vid(img, bboxes, color=(0, 0, 255), thick=3):
    '''
    Assumes bboxes format of [x1, y1, x2, y2]
    '''

    imcopy = np.copy(img)

    # Perform heatmap thresholding on bboxes, and append to queue
    reduced_bboxes_ = reduced_bboxes(bboxes, img.shape, threshold=1)
    bbox_queue.append(reduced_bboxes_)
    
    # Unpack all bboxes in queue to a list and again perform heatmap thresholding
    my_list = [item for sublist in list(bbox_queue) for item in sublist]
    reduced_queue = reduced_bboxes(my_list, img.shape, threshold=7)

    # Go through each of the reduced_bboxes and draw rectangle if large enough
    for bbox in reduced_queue:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        if ((x2-x1)*(y2-y1)) > 3670:
            cv2.rectangle(imcopy, (x1, y1), (x2, y2), color, thick)
    
    return imcopy


def draw_boxes_img(img, bboxes, color=(0, 0, 255), thick=3):
    '''
    Assumes bboxes format of [x1, y1, x2, y2]
    '''
    
    imcopy = np.copy(img)
    
    # Perform heatmap thresholding on bboxes
    reduced_bboxes_ = reduced_bboxes(bboxes, img.shape, threshold=1)

    # Go through each of the reduced_bboxes and draw rectangle
    for bbox in reduced_bboxes_:
        cv2.rectangle(imcopy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)
    
    return imcopy
