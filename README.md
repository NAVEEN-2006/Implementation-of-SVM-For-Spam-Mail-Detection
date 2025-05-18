# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

## Program:
Program to implement the SVM For Spam Mail Detection.

Developed by: NAVEEN KUMAR S

Register Number: 212223040129
```python
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/61ed7aa4-c7e9-4475-a943-fb6c029be76e)

![image](https://github.com/user-attachments/assets/fe243bd5-51cc-407e-9603-f71398d2afab)

![image](https://github.com/user-attachments/assets/819d2e78-d8f4-468b-aa2b-a29b2f76f6ad)

![image](https://github.com/user-attachments/assets/2f91d5d4-16b8-4070-8cc6-e805d36ccc93)

![image](https://github.com/user-attachments/assets/8f0d185b-1f77-4f1b-b810-e0f21f8bf266)

![image](https://github.com/user-attachments/assets/3e43e2eb-acb5-49da-a98a-10d977d620da)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
