import pandas as pd
import numpy as np
data = pd.read_csv("DigitalAd_dataset.csv")
print(data.shape)
print(data.head(5))
X = data.iloc[:, :-1].values  # All columns except the last one (features)
Y = data.iloc[:, -1].values   # Last column as target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler		# Feature scaling (standardizing the features)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression		# Train a Logistic Regression model
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)				# Make predictions on the test set
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)			# Evaluate the model using confusion matrix
print("Confusion Matrix: ")
print(cm)
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred) * 100))
age = int(input("Enter New Customer Age: "))		# Predict for a new customer
sal = int(input("Enter New Customer Salary: "))
newCust = [[age, sal]]
result = model.predict(sc.transform(newCust))		# Output the prediction result
print(result)
if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")
