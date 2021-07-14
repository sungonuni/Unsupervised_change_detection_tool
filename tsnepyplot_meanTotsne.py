from collections import defaultdict
from re import I
import sys
import numpy as np
import os
from numpy.core.fromnumeric import shape
from numpy.core.shape_base import block
from tqdm import tqdm
from skimage import io
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def mean_and_tsne(img1, img2, gt):
    assert img1.shape == img2.shape
    block_size = 3
    uc_element = np.empty((0,54), float)
    c_element = np.empty((0,54), float)

    gt = gt.reshape(-1,1)

    img1=img1.astype(float)
    img1_padded = np.pad(
        img1, ((block_size//2, block_size//2), (block_size//2, block_size//2), (0, 0)), 'constant', constant_values=(0))

    img2=img2.astype(float)
    img2_padded = np.pad(
        img2, ((block_size//2, block_size//2), (block_size//2, block_size//2), (0, 0)), 'constant', constant_values=(0))

    # pixel_diff = np.absolute(img1_padded-img2_padded)
    pixel_concat = np.concatenate((img1_padded, img2_padded), axis=2)

    concat_conv_vectors = np.zeros(
        (img1.shape[0]*img1.shape[1], block_size*block_size*pixel_concat.shape[-1]))
    
    # 3x3 sliding window coping
    for r_idx in range(img1.shape[0]):
        for c_idx in range(img1.shape[1]):
            concat_conv_vector = pixel_concat[r_idx: r_idx+block_size,
                                          c_idx:c_idx+block_size].reshape(-1)
            concat_conv_vectors[r_idx*img1.shape[1] + c_idx] = concat_conv_vector

    # Matching pixel one by one with gt, save respectively
    uc_indices = np.where(gt == 0)
    uc_element = concat_conv_vectors[uc_indices[0], :]

    c_indices = np.where(gt != 0)
    c_element = concat_conv_vectors[c_indices[0], :]

    if uc_element.size == 0:
        uc_element = np.zeros((2,54),float)
    elif c_element.size == 0:
        c_element = np.zeros((2,54),float)

    uc_element_mean = np.expand_dims(np.mean(uc_element, axis=0), axis=0)
    c_element_mean = np.expand_dims(np.mean(c_element, axis=0), axis=0)

    # Get means and do t-sne
    combine_mean = np.concatenate((uc_element_mean,c_element_mean), axis=0)
    combine_reduce = TSNE(n_components=3, random_state=0).fit_transform(combine_mean)
    uc_element_reduce = np.expand_dims(combine_reduce[0, :], axis=0)
    c_element_reduce = np.expand_dims(combine_reduce[1, :], axis=0)

    return uc_element_reduce, c_element_reduce

def mk_3dscatter(dataset, uc_total, c_total):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(c_total.shape[0]):
        x_c = c_total[i,0]
        y_c = c_total[i,1]
        z_c = c_total[i,2]
        ax.scatter(x_c, y_c, z_c, c='b', marker='o')

    for i in range(uc_total.shape[0]):
        x_uc = uc_total[i,0]
        y_uc  = uc_total[i,1]
        z_uc  = uc_total[i,2]
        ax.scatter(x_uc, y_uc, z_uc, c='r', marker='^')

    ax.set_title(dataset)
    ax.set_xlabel('X Label')
    #ax.set_xlim3d(-1500, 1500)
    ax.set_ylabel('Y Label')
    #ax.set_ylim3d(-500, 500)
    ax.set_zlabel('Z Label')
    #ax.set_zlim3d(-200, 200)
    
    ax.view_init(0,0)
    plt.savefig(dataset+'_figure_meanTotsne_YZ.png')
    ax.view_init(0,90)
    plt.savefig(dataset+'_figure_meanTotsne_XZ.png')
    ax.view_init(90,90)
    plt.savefig(dataset+'_figure_meanTotsne_XY.png')
    print('figure saved')

if __name__ == "__main__":
    
    # parameter
    preloading = False
    dataset = 'SCD512'

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
        uc_total = np.load('./'+dataset+'_uc_total.npy')
        c_total = np.load('./'+dataset+'_c_total.npy')

        mk_3dscatter(dataset, uc_total, c_total)

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
            uc_total = np.empty((0,3), float)
            c_total = np.empty((0,3), float)

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
                uc_1img, c_1img = mean_and_tsne(img_pre, img_post, img_gt)
                
                #print(uc_1img.shape)    #(1,3)
                #print(c_1img.shape)     #(1,3)
                uc_total = np.append(uc_total, uc_1img, axis=0)
                c_total = np.append(c_total, c_1img, axis=0)

            print(uc_total.shape)
            print(c_total.shape)

            # Save the total array
            np.save('./'+dataset+'_uc_total', uc_total)
            np.save('./'+dataset+'_c_total', c_total)
            print('C, UC array saved')

            mk_3dscatter(dataset, uc_total, c_total)
            

            
            
            
            




