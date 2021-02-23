# Machine Learning Course

## Assignment 1 --> K-Nearest-Neighbor:

- Write a python code without using any library, for a small dataset (use anything) with 10 samples, 2-d space, and two classes (0,1).
- Read a sample input (x,y) from the user.
- Read k from the user.
- Find and print the k nearest neighbor.

## Assignment 2 --> K-Mean-Clustering:

- breast_data.csv is Wisconsin Diagnostic Breast Cancer dataset. Data has D = 30 features and N = 569 samples.
- Load this data and run K-means clustering on this data using python.
  **[Here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)** is the reference to the k-mean function that you should use.

- The file breast_truth.csv contains 569 vectors each has 0/1 indicating the true clustering of the dataset (0 = benign, 1 = malign). Try different initial values and tell me which initial values led to the best
  accuracy? Show your code.

- **Bounus** : Under which number of clusters (K) can you reach better accuracy, show your code.

Notes: Upload only one .py file for all the answers.

### Reference:

O. Mangasarian, W. Street and W. Wolberg, Breast cancer diagnosis and prognosis via linear pro-
gramming, Operations Research, 1995. Dataset available at http://pages.cs.wisc.edu/~olvi/uwmp/
cancer.html#diag
