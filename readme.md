# DistanceClassifier

A classification model that takes in labeled training data and makes class assignments based upon euclidean distance from the centroids of each class in the training data. This model is ideally suited for use with data generated from unsupervised learning methods such as k-means clustering.

This model mimics the sklearn API and is compatible with sklearn infrastructure for data pipelining and model selection. Example usage can be found below.

```python
import DistanceClassifier

# instantiate model class
DC = DistanceClassifier()
# train model
DC.fit(X_train, y_train)
# make class predictions on test data
y_pred = DC.predict(X_test)
```
