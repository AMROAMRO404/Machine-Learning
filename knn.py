# kNN Algorithm, assignment 1 for Machine Learning course
# Author : Amro Amro


# each item in the dataset has x-coordinate, y-coordinate and class (0 or 1)
dataset = [
    [7.5, 5, 1],
    [8, 5.5, 1],
    [4, 3, 0],
    [4, 4, 0],
    [3, 4, 0],
    [6.5, 3, 1],
    [7, 5, 1],
    [4.5, 4.5, 0],
    [5, 5, 0],
    [7, 4, 1]
]

k = int(input("Enter the k value : "))
x_coordinate = float(input("Enter the x value : "))
y_coordinate = float(input("Enter the y value : "))

# this function to calculate distance between two points :
# the first is (x_coordinate, y_coordinate)
# the seconde is each point in dataset

distances = []
sorted_distances = []


def calculate_distancees(x, y):
    euclidean_distance = ((x_coordinate-x)**2 + (y_coordinate-y)**2)**0.5
    return euclidean_distance


def k_nearest_neighbors(x_coordinate, y_coordinate, k):
    for i in range(len(dataset)):
        distances.append(
            [calculate_distancees(dataset[i][0], dataset[i][1]), i, dataset[i][2]])

    distances.sort(key=lambda x: x[0])
    class0 = 0
    class1 = 0
    for i in range(k):
        if distances[i][2] == 0:
            class0 += 1
        else:
            class1 += 1
    if class0 > class1:
        print("this point is classified to be in the class 0")
        return (x_coordinate, y_coordinate, 0)
    else:
        print("this point is classified to be in the class 1")
        return (x_coordinate, y_coordinate, 1)


my_point = k_nearest_neighbors(x_coordinate, y_coordinate, k)
print(my_point)
