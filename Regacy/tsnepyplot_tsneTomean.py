from collections import defaultdict
from re import I
import sys
import math
import numpy as np
import os
import pickle
from numpy.core.fromnumeric import shape
from numpy.core.shape_base import block
from tqdm import tqdm
from skimage import filters
from skimage import io
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def pca_change_classification(img1, img2, gt):
    assert img1.shape == img2.shape
    block_size = 3
    uc_element = np.empty((0,3), float)
    c_element = np.empty((0,3), float)

    gt = gt.reshape(-1,1)

    img1=img1.astype(float)
    img1_padded = np.pad(
        img1, ((block_size//2, block_size//2), (block_size//2, block_size//2), (0, 0)), 'constant', constant_values=(0))

    img2=img2.astype(float)
    img2_padded = np.pad(
        img2, ((block_size//2, block_size//2), (block_size//2, block_size//2), (0, 0)), 'constant', constant_values=(0))

    # pixel_diff = np.absolute(img1_padded-img2_padded)
    pixel_concat = np.concatenate((img1_padded, img2_padded), axis=2)

    diff_conv_vectors = np.zeros(
        (img1.shape[0]*img1.shape[1], block_size*block_size*pixel_concat.shape[-1]))
    
    # 3x3 sliding window coping
    for r_idx in range(img1.shape[0]):
        for c_idx in range(img1.shape[1]):
            diff_conv_vector = pixel_concat[r_idx: r_idx+block_size,
                                          c_idx:c_idx+block_size].reshape(-1)
            diff_conv_vectors[r_idx*img1.shape[1] + c_idx] = diff_conv_vector

    # Conduct t-sne
    features = TSNE(n_components=3, random_state=0).fit_transform(diff_conv_vectors)

    # Matching pixel one by one with gt, save respectively
    uc_indices = np.where(gt == 0)
    uc_element = features[uc_indices[0], :]

    c_indices = np.where(gt != 0)
    c_element = features[c_indices[0], :]
    
    return uc_element, c_element

def dict_per_img(imgnum, uc_dict_total, c_dict_total):
    uc_total_sum = np.zeros((1,3), float)
    c_total_sum = np.zeros((1,3), float)
    uc_dict_per_img = np.empty((0,3), float)
    c_dict_per_img = np.empty((0,3), float)
    num_of_uc_elem = 0
    num_of_c_elem = 0

    for idx in tqdm(range(imgnum)):
        # Take mean dimension-wise to make 1 vector per image and stack.
        uc_dict_per_img = np.append(uc_dict_per_img, np.mean(uc_dict_total[idx], axis=1), axis=0) # uc_dict_total[idx] (1,64434,3)
        c_dict_per_img = np.append(c_dict_per_img, np.mean(c_dict_total[idx], axis=1), axis=0)

        # Sum all element value dimension-wise and number of elements for get total mean. 
        uc_total_sum += np.sum(uc_dict_total[idx], axis=1) # test required
        c_total_sum += np.sum(c_dict_total[idx], axis=1)
        num_of_uc_elem += np.array(uc_dict_total[idx]).shape[1]
        num_of_c_elem += np.array(c_dict_total[idx]).shape[1]
            
    # Get total elements means (Centroid of each cluster)
    center_of_total_uc = uc_total_sum / num_of_uc_elem
    center_of_total_c = c_total_sum / num_of_c_elem

    print(uc_dict_per_img.shape)
    print(c_dict_per_img.shape)
    return center_of_total_uc, center_of_total_c, uc_dict_per_img, c_dict_per_img    

def mk_3dscatter(dataset, center_of_total_uc, center_of_total_c, uc_dict_per_img, c_dict_per_img):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(c_dict_per_img.shape[0]):
        x_c = c_dict_per_img[i,0]
        y_c = c_dict_per_img[i,1]
        z_c = c_dict_per_img[i,2]
        ax.scatter(x_c, y_c, z_c, c='b', marker='o')

    for i in range(uc_dict_per_img.shape[0]):
        x_uc = uc_dict_per_img[i,0]
        y_uc  = uc_dict_per_img[i,1]
        z_uc  = uc_dict_per_img[i,2]
        ax.scatter(x_uc, y_uc, z_uc, c='r', marker='^')

    
    ax.scatter(center_of_total_uc[0,0], center_of_total_uc[0,1], center_of_total_uc[0,2], c='g', marker='^')
    ax.scatter(center_of_total_c[0,0], center_of_total_c[0,1], center_of_total_c[0,2], c='y', marker='o')

    ax.set_title(dataset)
    ax.set_xlabel('X Label')
    #ax.set_xlim3d(-1500, 1500)
    ax.set_ylabel('Y Label')
    #ax.set_ylim3d(-500, 500)
    ax.set_zlabel('Z Label')
    #ax.set_zlim3d(-200, 200)
    
    ax.view_init(0,0)
    plt.savefig(dataset+'_figure_3d_scatter_YZ.png')
    ax.view_init(0,90)
    plt.savefig(dataset+'_figure_3d_scatter_XZ.png')
    ax.view_init(90,90)
    plt.savefig(dataset+'_figure_3d_scatter_XY.png')
    print('figure saved')

def cnt_len(dict):
    count = 0
    for i in enumerate(dict):
        count += 1
    return count

if __name__ == "__main__":
    
    # parameter
    preloading = False
    dataset = 'SCD128'

    # Dataset dir (changing required)
    cdd128 = [
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/CDData128/test/A',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/CDData128/test/B',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/CDData128/test/OUT'
        ]
    cdd256 = [
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/CDData/subset/test/A',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/CDData/subset/test/B',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/CDData/subset/test/OUT'
        ]
    
    scd128 = [
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData128/test/A',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData128/test/B',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData128/test/OUT'
    ]

    scd256 = [
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData256/test/A',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData256/test/B',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData256/test/OUT'
    ]

    scd512 = [
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData/test/A',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData/test/B',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/SCData/test/OUT'
    ]

    test_dir = [
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/test/A',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/test/B',
        '/data/workspace/seonggon/ev21_phase2/SNUnet/datasets/test/OUT',
    ]

    if preloading:
        print("Load presaved data")
        with open('./'+dataset+'_uc_dict_total.pkl', 'rb') as f:
            uc_dict_total = pickle.load(f)
        with open('./'+dataset+'_c_dict_total.pkl', 'rb') as f:
            c_dict_total = pickle.load(f)

        # check equality of uc_dict_total and c_dict_total 
        ucimgnum = cnt_len(uc_dict_total)
        cimgnum = cnt_len(c_dict_total)
        if ucimgnum != cimgnum:
            print('ucimgnum and cimgnum doesnt match!')
            sys.exit()

        center_of_total_uc, center_of_total_c, uc_dict_per_img, c_dict_per_img = dict_per_img(ucimgnum, uc_dict_total, c_dict_total)
        mk_3dscatter(dataset, center_of_total_uc, center_of_total_c, uc_dict_per_img, c_dict_per_img)

    else:
        print('dataset: '+dataset)
        if dataset == 'CDD256':
            img_pre_path = cdd256[0]
            img_post_path = cdd256[1]
            img_gt_path = cdd256[2]
        elif dataset == 'CDD128':
            img_pre_path = cdd128[0]
            img_post_path = cdd128[1]
            img_gt_path = cdd128[2]
        elif dataset == 'SCD128':
            img_pre_path = scd128[0]
            img_post_path = scd128[1]
            img_gt_path = scd128[2]
        elif dataset == 'SCD256':
            img_pre_path = scd256[0]
            img_post_path = scd256[1]
            img_gt_path = scd256[2]
        elif dataset == 'SCD512':
            img_pre_path = scd512[0]
            img_post_path = scd512[1]
            img_gt_path = scd512[2]
        elif dataset == 'test':
            img_pre_path = test_dir[0]
            img_post_path = test_dir[1]
            img_gt_path = test_dir[2]    
        else:
            print("No dataset found!")
            sys.exit()

        img_pre_list = [i for i in os.listdir(img_pre_path)]
        img_pre_list.sort()

        img_post_list = [i for i in os.listdir(img_post_path)]
        img_post_list.sort()

        img_gt_list = [i for i in os.listdir(img_gt_path)]
        img_gt_list.sort()


        if len(img_pre_list) != len(img_post_list):
            print("Can't make CD image: pre and post images don't match.")
            sys.exit()
        else:
            uc_dict_total = defaultdict(list)
            c_dict_total = defaultdict(list)
            uc_dict_per_img = np.empty((0,3), float)
            c_dict_per_img = np.empty((0,3), float)
            
            # imgnum = len(img_pre_list)
            if len(img_pre_list) > 500:
                imgnum = 500
            else:
                imgnum = len(img_pre_list)

            # Stack the total element of dataset per each cluster in order of image
            print("Stack the total element of dataset per each cluster in order of image")
            for idx in tqdm(range(imgnum)):
                img_pre = io.imread(os.path.join(img_pre_path, img_pre_list[idx]))
                img_post = io.imread(os.path.join(img_post_path, img_post_list[idx]))
                img_gt = io.imread(os.path.join(img_gt_path, img_gt_list[idx]))
                uc_ndarray, c_ndarry = pca_change_classification(img_pre, img_post, img_gt)
                
                #print(uc_ndarray.shape)    #(52434,3)
                #print(c_ndarry.shape)      #(13102,3)
                uc_dict_total[idx].append(uc_ndarray)
                c_dict_total[idx].append(c_ndarry)

            # Save the dict
            with open('./'+dataset+'_uc_dict_total.pkl', 'wb') as f:
                pickle.dump(uc_dict_total, f)
            with open('./'+dataset+'_c_dict_total.pkl', 'wb') as f:
                pickle.dump(c_dict_total, f)
            print('C, UC dict saved')

            center_of_total_uc, center_of_total_c, uc_dict_per_img, c_dict_per_img = dict_per_img(imgnum, uc_dict_total, c_dict_total)

            mk_3dscatter(dataset, center_of_total_uc, center_of_total_c, uc_dict_per_img, c_dict_per_img)
            

            
            
            
            




