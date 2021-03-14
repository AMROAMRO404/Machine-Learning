from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

iris = load_iris()
X = iris['data']
Y = iris['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

# scaling the features
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


number_of_neurons = 5
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(number_of_neurons),
                    max_iter=3000)
mlp.fit(X_train, Y_train)
nn_pridictions = mlp.predict(X_test)
print(
    f'\nThe nn score for {number_of_neurons} neurons = {mlp.score(X_test, Y_test)}\n')
# print(nn_pridictions)


# study the overfit as we increase the hidden neurons
to_study_accurecy = []
for i in range(10):
    mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(i+1),
                        max_iter=3000)
    mlp.fit(X_train, Y_train)
    to_study_accurecy.append(mlp.score(X_test, Y_test))

# cross validation
print("3 fold cross validation : ")
scores = cross_val_score(mlp, X, Y, cv=3)
print(scores)

xpoints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(xpoints, to_study_accurecy, marker='o')
plt.xlabel('number of hidden neurons')
plt.ylabel('accuracy')
plt.show()
