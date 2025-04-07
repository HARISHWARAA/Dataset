# -*- coding: utf-8 -*-
"""Wild Fire Risk Prediction.ipynb
Original file is located at
    https://colab.research.google.com/drive/15k4FzsuyW7GtSOGMA0xYuCjhiY2vCRP_
"""

# impoting necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/HARISHWARAA/Dataset/refs/heads/main/wildfire_risk_dataset.csv')

df.head()

df.shape

df.info()

df.isnull().sum()

sns.countplot(x='fire_risk', data=df)
plt.title("Fire Risk Class Distribution")
plt.show()

sns.histplot(df['temperature'], kde=True)
plt.title("Temperature Distribution")
plt.show()

X = df.drop('fire_risk', axis=1)
y = df['fire_risk']

# Importing necessary libraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluating model performance with accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
