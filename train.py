import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

os.makedirs("artifacts", exist_ok=True)

try:
    data = pd.read_csv('iris.csv')
except FileNotFoundError:
    print("iris.csv not found, creating a dummy one for demonstration.")
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    iris.data.columns = ['sepal_length','sepal_width','petal_length','petal_width']
    iris.target_names = np.array(['setosa', 'versicolour', 'virginica'])
    iris.target = iris.target.map({0: 'setosa', 1: 'versicolour', 2: 'virginica'})
    iris_df = pd.concat([iris.data, iris.target.rename('species')], axis=1)
    iris_df.to_csv('iris.csv', index=False)
    data = pd.read_csv('iris.csv')


train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
accuracy=metrics.accuracy_score(prediction,y_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy))

joblib.dump(mod_dt, "artifacts/model.joblib")

metrics_data = {
    'metric': ['accuracy'],
    'value': [accuracy]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv("metrics.csv", index=False)

print("Model trained, saved to artifacts/model.joblib, and metrics saved to metrics.csv")
