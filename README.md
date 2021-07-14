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
