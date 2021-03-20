import matplotlib.pyplot as plt
# dataset
X = [
    [1, 5, 1],
    [2, 3, 1],
    [3, 2, -1],
    [4, 6, -1],
    [4, 7, 1],
    [5, 9, 1],
    [6, 5, -1],
    [6, 7, 1],
    [8, 5, -1],
    [8, 8, -1]
]
to_plot_positive_samples = []
for i in range(len(X)):
    if X[i][2] >= 0:
        to_plot_positive_samples.append([X[i][0], X[i][1]])
to_plot_negative_samples = []
for i in range(len(X)):
    if X[i][2] < 0:
        to_plot_negative_samples.append([X[i][0], X[i][1]])
print(to_plot_negative_samples)
print(to_plot_positive_samples)
# algorithm logic :

# initialization step: for each example X, we need to set D(x) = 1/n , D: sample weights, n:number of samples
# sample_weights = []
# for i in range(len(X)-1):
#     sample_weights[i] = 1/len(X)

# iteration step
# 1- we need to find the best classifier h(x) using weights D(x), where h: week classifier
# 2- compute the error rate ɛ as:
#       ɛ = Σi=0 to n   D(x sub i) * I[y sub i != h(x sub i)]
# 3- assign weight 	α to classifier h in the final hypothesis
# 4- for each x sub i, D(x sub i) =  D(x sub i) * exp(α * I[y sub i != h(x sub i)])
# 5- normalize D(x sub i) so that Σi=0 to n D(x sub i) = 1
week_classifiers = []
x11 = [to_plot_positive_samples[i][0]
       for i in range(len(to_plot_positive_samples))]
x21 = [to_plot_positive_samples[i][1]
       for i in range(len(to_plot_positive_samples))]
x12 = [to_plot_negative_samples[i][0]
       for i in range(len(to_plot_negative_samples))]
x22 = [to_plot_negative_samples[i][1]
       for i in range(len(to_plot_negative_samples))]
plt.scatter(x12, x22, marker='_', s=300, c='red')
plt.scatter(x11, x21, marker='+', s=300, c='blue')
plt.show()
