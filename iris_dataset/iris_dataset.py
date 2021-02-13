
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors = 1)

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)


def execute():
    knn.fit(X_train, y_train)
    knn_test_score = knn.score(X_test, y_test)
    print('KNN TEST SCORE: {:.2f}'.format(knn_test_score))


if __name__ == '__main__':
    execute()
