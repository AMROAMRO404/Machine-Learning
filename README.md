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
gramming, Operations Research, 1995. Dataset available **[Here](http://pages.cs.wisc.edu/~olvi/uwmp/cancer.html#diag)**

## Assignment 3 --> Neural Network:

Write a program that uses the neural network in scikit, **[Here](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)**

your code should do the following

- read the iris data
- establish a neural network with one hidden layer.
- use 3 fold cross-validation
- study the overfit as we increase the hidden neurons
- plot the results of the (number of hidden neurons vs the accuracy)

## Assignment 4 --> Adaboost:

Implement adaboost without relyting on scikit-learn :

- Use the same data and the same weak classifiers as in the lecture slides.
- Show the values of alpha for each classifier.
- Show the class of one random sample.
