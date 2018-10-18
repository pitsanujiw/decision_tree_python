from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#  Import data_sets

iris = load_iris()
data = iris['data']
target = iris['target']

# Training datasets
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.4)

mlp = MLPClassifier(hidden_layer_sizes=(
    100,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000, tol=0.0001)

# Training models
mlp.fit(x_train, y_train)

# Test models
predictions = mlp.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)
print(classification_report(y_test, predictions))


