## Instruction
This is official code of ['PC2VA : : Principal Component and Change Vector Analysis for Unsupervised Change Detection in satellite images'](https://drive.google.com/file/d/1LUXjD6Ubswfe6bBrtV3DZxLYVt1W8pqV/view?usp=sharing). 
You can perform the unsupervised change detection at bi-temporal sattlelite imagary in PC2VA method.


## Description
Change detection of temporal remote sensing images was mainly performed in a supervised manner after the elevation of deep learning.
However, the necessity for unsupervised change detection emerged because of the enormous time and resources to produce the well refined labeled change detection datasets. 
To address this issue, unsupervised method is required, however due to lack of exploration of unsupervised deep learning methods, conventional based unsupervised methods are still on the mainstream. 

We propose principal component change vector analysis (PC2VA), which predicts the change information by using decision making method on the globally normalized norm of local feature vector. 
Local feature vector is computed by projecting flattened local patch to the selected eigenvectors of global region. 
After extracting feature vector of every patch is finished, we exploit change vector analysis(CVA) to compute the magnitude of local feature vector and do normalize the norm at the global region. 
Finally, K-means clustering classifies normalized magnitude into changed information or unchanged information. 
This method outperforms the unsupervised deep learning methods on the DynamicEarthNet Challenge unsupervised binary land cover change detection dataset

## Requirements
- Python 3.8
- UMAP
- scikit-learn
- matpyplot

## Dataset
- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) (Change Detection Dataset)
- [SCD](https://drive.google.com/file/d/1cAyKCjRiRKfTysX1OqtVs6F1zbEI0EGj/view?usp=sharing) (SECOND CD Dataset)

## PC^2VA unsupervised change detection

    python main.py

## Get t-sne visualization of dataset (compute mean before t-sne)

    python tsnepyplot_meanTotsne.py
 
## Get t-sne visualization of dataset (compute t-sne before mean)

    python tsnepyplot_tsneTomean.py
    
## Get UMAP visualization of dataset (compute UMAP before mean)

    python umappyplot.py
