import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikacji: {:.2f}".format(accuracy))

accuracies = []
for n in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.plot(range(1, 11), accuracies)
plt.xlabel("Liczba sąsiadów")
plt.ylabel("Dokładność")
plt.show()

N = 5

knn = KNeighborsClassifier(n_neighbors=N)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikacji dla N = {}: {:.2f}".format(N, accuracy))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Macierz konfuzji:")
print(cm)

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.xlabel("Długość korzeni")
plt.ylabel("Szerokość korzeni")
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred)
ax.set_xlabel("Długość korzeni")
ax.set_ylabel("Szerokość korzeni")
ax.set_zlabel("Długość płatka")
plt.show()