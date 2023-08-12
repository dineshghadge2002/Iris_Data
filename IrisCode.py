#importing Libraries
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler

#uploading dataset
dataset=pd.read_csv("IRIS (1).csv")
print(dataset.columns)

# Exploratory Data Analysis (EDA)

# Import required libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display pair plot to visualize relationships between features
sns.pairplot(dataset, hue='species', markers=["o", "s", "D"])

# Display correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Display box plots to visualize feature distributions
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataset, orient='h')
plt.title("Feature Distributions")
plt.show()

# Display count plot of species
plt.figure(figsize=(6, 4))
sns.countplot(data=dataset, x='species')
plt.title("Species Distribution")
plt.show()

# Display violin plots to visualize feature distributions across species
plt.figure(figsize=(12, 6))
sns.violinplot(data=dataset, x='species', y='sepal_length')
plt.title("Feature Distribution across Species")
plt.show()

#label Encoding
le = LabelEncoder()
dataset['species']=le.fit_transform(dataset['species'])
print(dataset)

#feature/column selection
x = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = dataset[['species']]

#training model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split( x, y, test_size=0.20, random_state=4)
model=LogisticRegression()
model.fit(x_train,y_train)

#co-efficient
print(model.coef_)

#intercept
print(model.intercept_)

y_pred=model.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

#accuracy score
print(model.score(x_test,y_test))
