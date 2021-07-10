# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

## Data Collection and Processing

## loading the data from csv fie to pandas dataframne

car_dataset=pd.read_csv('car data.csv')

car_dataset.head()


## Checking the nummber of rows and columns

car_dataset.shape


#Getting some info about the dataset

car_dataset.info()


## Checking the number of missing values

car_dataset.isnull().sum()


## Checking the distribution of categorical data

print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

## Encoding the categorical data

#encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)


#encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

#encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


car_dataset.head()


##Splitting the data and Target

X= car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y=car_dataset['Selling_Price']

print(X)
print(Y)

## Splitting Training and Test Data

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.1, random_state=2)


# Model Training

# 1)Linear Regression

# loading the linear regression model

lin_reg_model=LinearRegression()

lin_reg_model.fit(X_train,Y_train)


# Model Evaluation

# Prediction on Training Data

training_data_prediction=lin_reg_model.predict(X_train)


#R squared Error

error_score=metrics.r2_score(Y_train,training_data_prediction)
print("R squared Error:",error_score)


## Visualise the actual prices adn Predicted Prices

plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Prediction on Test Data

test_data_prediction=lin_reg_model.predict(X_test)

#R squared Error

error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R squared Error:",error_score)


plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()