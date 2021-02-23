from sklearn.cluster import KMeans
import csv
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv(r'Assignment 2/breast_data.csv')
truth_data = pd.read_csv(r'Assignment 2/breast_truth.csv')

# Load this data and run K-means clustering on this data using python.
kmeans = KMeans(n_clusters=2, random_state=2).fit(data)
# test accuresy
print(accuracy_score(truth_data, kmeans.labels_))
# Try different initial values and tell me which initial values led to the best accuracy? Show your code
test_random_state_accurecy = []
print("change #random_state")
for i in range(20):
    kmeans = KMeans(n_clusters=2, random_state=i).fit(data)
    test_random_state_accurecy.append(
        accuracy_score(truth_data, kmeans.labels_))
print(f'the best accuresy = {max(test_random_state_accurecy)} at random_state = {test_random_state_accurecy.index(max(test_random_state_accurecy)) +1}')


# Under which number of clusters (K) can you reach better accuracy, show your code.
print("change #clusters")
test_n_clusters_accurecy = []
for i in range(20):
    kmeans = KMeans(n_clusters=i+1, random_state=2).fit(data)
    test_n_clusters_accurecy.append(
        accuracy_score(truth_data, kmeans.labels_))
print(
    f'the best accuresy = {max(test_n_clusters_accurecy)}  at n_clusters = {test_n_clusters_accurecy.index(max(test_n_clusters_accurecy)) +1}')
