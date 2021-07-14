import numpy as np
import math
 
import torch
from sklearn.cluster import KMeans
from saturateSomePercentile import saturateImage
 
# https://github.com/rulixiang/ChangeDetectionPCAKmeans/blob/master/pca_kmeans.m
# https://ieeexplore.ieee.org/document/5196726
 
def detect_change(img1: np.ndarray, img2: np.ndarray, block_size: int, num_comp: int) -> np.ndarray:
    assert img1.shape == img2.shape

    img1_padded = np.pad(
        img1, ((block_size//2, block_size//2), (block_size//2, block_size//2), (0, 0)), 'constant', constant_values=(0))
 
    img2_padded = np.pad(
        img2, ((block_size//2, block_size//2), (block_size//2, block_size//2), (0, 0)), 'constant', constant_values=(0))

    # Generate difference map
    pixel_diff = np.absolute(img1_padded-img2_padded)

    # N X N X C patching 
    diff_conv_vectors = np.zeros(
        (img1.shape[0]*img1.shape[1], block_size*block_size*img1.shape[-1]))
    for r_idx in range(img1.shape[0]):
        for c_idx in range(img1.shape[1]):
            diff_conv_vector = pixel_diff[r_idx: r_idx+block_size,
                                          c_idx:c_idx+block_size].reshape(-1)
            diff_conv_vectors[r_idx*img1.shape[1] + c_idx] = diff_conv_vector
 
    # Normalization
    normalized_vectors = (
        diff_conv_vectors - np.mean(diff_conv_vectors, axis=0)) / np.std(diff_conv_vectors, axis=0)

    # Created feature map by PCA
    covariance = np.dot(normalized_vectors.T, normalized_vectors)
    _, eigen_vec = np.linalg.eig(covariance)
 
    features = np.dot(normalized_vectors, eigen_vec[:, :num_comp])

    # CVA
    detectedChangeMap=np.linalg.norm(features, axis=(1))
    detectedChangeMapNormalized = (detectedChangeMap-np.amin(detectedChangeMap))/(np.amax(detectedChangeMap)-np.amin(detectedChangeMap))
    detectedChangeMapNormalized2d = detectedChangeMapNormalized[:, np.newaxis]
    
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(detectedChangeMapNormalized2d)
 
    change_map = np.reshape(
        kmeans.labels_, (img1.shape[0], img1.shape[1]))
 
    change_map = change_map * 255 

    # Cluster Correction
    if len(change_map[change_map==255]) > len(change_map[change_map==0]):
        change_map = np.where(change_map==0, 10, change_map)
        change_map = np.where(change_map==255, 0, change_map)
        change_map = np.where(change_map==10, 255, change_map)

    return change_map