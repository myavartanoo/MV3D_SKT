import numpy        as np
import cv2
import os
import glob

from collections        import defaultdict
from warnings           import warn
from time               import time
from config_rpn         import cfg



# get all file names starts from a prefix string.
def get_file_names(data_dir, data_type, driver, date, index=None):
    dir_path = os.path.join(data_dir, data_type)
    driver_path = os.path.join(dir_path, date, driver)
    if index is None:
        prefix = driver_path + '/*'
        driver_files = glob.glob(prefix)
    else:
        prefix = [os.path.join(driver_path, file_name) for file_name in index]
        driver_files = [glob.glob(i + '*')[0] for i in prefix]
    return driver_files

def check_preprocessed_data(data_seg, dates_to_drivers, is_testset=False):
    problem_driver =  defaultdict(list)
    right_driver = defaultdict(list)

    for date, drivers in dates_to_drivers.items():
        for driver in drivers:
            rgb_files = get_file_names(data_seg, "rgb", driver, date)
            top_files = get_file_names(data_seg, "top", driver, date)
            # front_files = get_file_names(data_seg, "front", driver, date)
            gt_labels_files = get_file_names(data_seg, "gt_labels", driver, date)
            gt_boxes3d_files = get_file_names(data_seg, "gt_boxes3d", driver, date)

            if is_testset:
                value_set = set([len(rgb_files), len(top_files)])
            else:
                value_set = set([len(rgb_files), len(top_files), len(gt_labels_files), len(gt_boxes3d_files)])
            if len(value_set) != 1:
                # print('date is here {} and driver here {}'.format(date, driver))
                problem_driver[date].append(driver)
            else:
                right_driver[date].append(driver)

    for key, value in right_driver.items():
        print("CORRECT!, date {{'{}':{}}} has same number of rgbs, tops, gt_labels or gt_boxes".format(key, value))

    if len(problem_driver.keys()) != 0:
        for key, value in problem_driver.items():
            warn("INCORRECT! date {{'{}':{}}} has different number of rgbs, tops, gt_labels or gt_boxes".format(key,
                                                                                                             value))
        raise ValueError('Check above warning info to find which date and driver data is incomplete. ')
    return True

class timer:
    def __init__(self):
        self.init_time = time()
        self.time_now = self.init_time

    def time_diff_per_n_loops(self):
        time_diff = time() - self.time_now
        self.time_now = time()
        return time_diff

    def total_time(self):
        return time() - self.init_time


def box3d_to_top_box (boxes3d):
    num = len(boxes3d)
    boxes_top = np.zeros((num, 4), dtype=np.float32)

    for n in range(num):
        box = boxes3d[n]
        x0 = box[0, 0]
        y0 = box[0, 1]
        x1 = box[1, 0]
        y1 = box[1, 1]
        x2 = box[2, 0]
        y2 = box[2, 1]
        x3 = box[3, 0]
        y3 = box[3, 1]

        xmin = min(x0, x1, x2, x3)
        xmax = max(x0, x1, x2, x3)
        ymin = min(y0, y1, y2, y3)
        ymax = max(y0, y1, y2, y3)

        boxes_top[n] = np.array([xmin, ymin, xmax, ymax])
    return boxes_top

def draw_image_top(view_top):
    image_top   = np.sum(view_top,axis=2)
    image_top   = image_top-np.min(image_top)
    div         = np.max(image_top)-np.min(image_top)
    image_top   = (image_top/div*255)
    image_top   = np.dstack((image_top, image_top, image_top)).astype(np.uint8)
    return image_top


def box3d_in_top_view(boxes3d):
    # what if only some are outside of the range, but majorities are inside.
    for i in range(8):
        if cfg.TOP_X_MIN<=boxes3d[i,0]<=cfg.TOP_X_MAX and cfg.TOP_Y_MIN<=boxes3d[i,1]<=cfg.TOP_Y_MAX:
            continue
        else:
            return False
    return True

def draw_rpn_proposal(image_top, proposals, proposal_scores, darker=0.75):

    img_rpn_nms = image_top.copy()*darker
    scores      = proposal_scores
    inds        = np.argsort(scores)
    for n in range(len(inds)):
        i       = inds[n]
        box     = proposals[i,1:5].astype(np.int)
        v       =254*(1-proposal_scores[i])+1
        color   = (0,v,v)
        cv2.rectangle(img_rpn_nms,(box[0], box[1]), (box[2], box[3]), color, 1)

    return img_rpn_nms


def get_top_feature_shape(top_shape, stride):

    return (top_shape[0]//stride, top_shape[1]//stride)

