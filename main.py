import os
import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pca_kmeans_dcva import detect_change

# Dataset dir (changing required)
img_pre_path = './A/'
img_post_path = './B/'

img_pre_list = [i for i in os.listdir(img_pre_path)]
img_pre_list.sort()

img_post_list = [i for i in os.listdir(img_post_path)]
img_post_list.sort()

if len(img_pre_list) != len(img_post_list):
    print("can't make CD image: pre and post images don't match.")
else:
    imgnum = len(img_pre_list)
    for idx in tqdm(range(imgnum)):
        totalCount = str(idx+1) + '/' + str(imgnum)
        img_pre = io.imread(os.path.join(img_pre_path, img_pre_list[idx]))
        img_post = io.imread(os.path.join(img_post_path, img_post_list[idx]))
        
        change_map = detect_change(img_pre, img_post, 3, 3)
        
        resultDirectory = './result/'
        if not os.path.exists(resultDirectory):
            os.makedirs(resultDirectory)
        cv2.imwrite(resultDirectory+img_pre_list[idx], change_map)