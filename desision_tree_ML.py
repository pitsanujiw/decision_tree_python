from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#  Import dataSet

iris = load_iris()
data = iris['data']
target = iris['target']


# Training datasets
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.2)
# Create model
algorithmmodels = DecisionTreeClassifier()
model = algorithmmodels.fit(x_train, y_train)
predict = model.predict(x_test)
print(predict)

confusion_data = confusion_matrix(y_test, predict)
accuracy_data = accuracy_score(y_test, predict)
classReport_table = classification_report(y_test, predict)


print(confusion_data)
print(accuracy_data)
print(classReport_table)
