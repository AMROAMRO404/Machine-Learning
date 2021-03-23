import numpy as np

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
Y = [1, 1, -1, -1, 1, 1, -1, 1, -1, -1]  # Output
# algorithm logic :
# initialization step: for each example X, we need to set D(x) = 1/n , D: sample weights, n:number of samples
# iteration step
# 1- we need to find the best classifier h(x) using weights D(x), where h: week classifier
# 2- compute the error rate ɛ as:
#       ɛ = Σi=0 to n   D(x sub i) * I[y sub i != h(x sub i)]
# 3- assign weight 	α to classifier h in the final hypothesis
# 4- for each x sub i, D(x sub i) =  D(x sub i) * exp(α * I[y sub i != h(x sub i)])
# 5- normalize D(x sub i) so that Σi=0 to n D(x sub i) = 1


# initialization step: for each example X, we need to set D(x) = 1/n , D: sample weights, n:number of samples
sample_weights = []  # D
number_of_samples = len(X)
w = 1 / number_of_samples
sample_weights = [w]*10
# week classifiers
H = [[1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
     [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
     [-1, -1, -1, -1, 1, 1, -1, 1, -1, 1]]

# Find the error rate for each classifer
classifiers_with_error_rate = []
for i in range(len(H)):
    number_of_errors = 0
    h = H[i]
    for j in range(number_of_samples):
        if h[j] != Y[j]:
            number_of_errors += 1
    classifiers_with_error_rate.append([i, number_of_errors])
# find best one classifier
classifiers_with_error_rate.sort(key=lambda x: x[1])
# 0 --> first classifier  1 --> second  2 --> third
print(classifiers_with_error_rate)

# initialization Alpha
alpha = [0] * len(H)

# Iteration setup
for t in range(len(H)):
    # note that best classifier in first index of classifiers_with_error_rate
    best_classifier_index = classifiers_with_error_rate[t][0]
    h = H[best_classifier_index]

    # calculate error rate
    E = 0
    for i in range(number_of_samples):
        E += sample_weights[i] * (h[i] != Y[i])

    # assign weight α to classifier h in the final hypothesis
    alpha[t] = np.log((1-E)/E)/2

    # for each x sub i, D(x sub i) =  D(x sub i) * exp(α * I[y sub i != h(x sub i)])
    for i in range(number_of_samples):
        sample_weights[i] = sample_weights[i] * \
            np.exp(alpha[t] * (h[i] != Y[i]))

    # normalize D(x sub i) so that Σi=0 to n D(x sub i) = 1
    normalize_factor = 2 * np.sqrt((1-E)*E)
    for i in range(number_of_samples):
        if Y[i] != h[i]:
            sample_weights[i] = sample_weights[i] * \
                np.exp(alpha[t])/normalize_factor
        else:
            sample_weights[i] = sample_weights[i] * \
                np.exp(-1*alpha[t])/normalize_factor

    print("weights")
    print(np.round(sample_weights, 2))
    print("error rate")
    print(np.round(E, 2))
    print("alpha")
    print(np.round(alpha[t], 2))
    print("\n")
# pridict the output for each class
x = int(input("enter the index for your sample:"))
first_classifier = H[classifiers_with_error_rate[0][0]]
second_classifier = H[classifiers_with_error_rate[1][0]]
third_classifier = H[classifiers_with_error_rate[2][0]]
H_final = np.sign(alpha[0]*first_classifier[x]+alpha[1] *
                  second_classifier[x]+alpha[2] *
                  third_classifier[x])
print(f"pridiction of sample {x} is {H_final} \n")


all_pridiction = []
for i in range(number_of_samples):
    H_final = np.sign(alpha[0]*first_classifier[i]+alpha[1] *
                      second_classifier[i]+alpha[2] *
                      third_classifier[i])
    all_pridiction.append(H_final)
print(all_pridiction)
