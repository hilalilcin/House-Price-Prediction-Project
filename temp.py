# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

df = pd.read_excel("dataset.xlsx")
df.to_csv('dataset.csv',index = False) # converting excel to csv


#Exploratory Data Analysis - EDA 
df = pd.read_csv('dataset.csv')
print(df.head())

df.shape
print(df.shape)

df.info()
print(df.info())

df.nunique()
print(df.nunique())

df.describe()
print(df.describe())

# Checking Missing Values
print(df.isnull().sum())

null_counts = df.isnull().sum()
print(null_counts)

df.dtypes
print(df.dtypes)

# Data Visualization
corr_matrix = df.corr()
plt.figure(figsize=(10, 10))  # 9x9 matris boyutuna uygun bir grafik boyutu

# Visualiasiton with heatmap
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Feautures')
plt.show()




df.columns
print(df.columns)


# Spliting target variable and independent variables
X = df[['Transaction date', 'House Age',
       'Distance from nearest Metro station (km)',
       'Number of convenience stores', 'latitude', 'longitude',
       'Number of bedrooms', 'House size (sqft)']]

y = df['House price of unit area']


# Splitting to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Transaction date sütununu tarih formatına dönüştürme
df['Transaction date'] = df['Transaction date'].apply(numeric_to_date)

# Yeni sütunlar ekleyerek yıl, ay, gün ve hafta bilgilerini çıkarma
df['Year'] = df['Transaction date'].dt.year
df['Month'] = df['Transaction date'].dt.month
df['Day'] = df['Transaction date'].dt.day
df['Weekday'] = df['Transaction date'].dt.weekday  # 0: Pazartesi, 6: Pazar

# Sonuçları görüntüleme
print(df[['Transaction date', 'Year', 'Month', 'Day', 'Weekday']].head())

# Yeni sütunlar ekleyerek yıl, ay, gün ve hafta bilgilerini çıkarma
df['Year'] = df['Transaction date'].dt.year
df['Month'] = df['Transaction date'].dt.month
df['Day'] = df['Transaction date'].dt.day
df['Weekday'] = df['Transaction date'].dt.weekday  # 0: Pazartesi, 6: Pazar

# Sonuçları görüntüleme
print(df[['Transaction date', 'Year', 'Month', 'Day', 'Weekday']].head())


















# Models Linear 
# Creating Models
lr_model = LinearRegression()

# Training Model
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Performance Metrics
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

print(f"Linear Regression - R²: {r2_lr}, MAE: {mae_lr}, RMSE: {rmse_lr}")



# Ridge Regression Modeli
ridge_model = Ridge(alpha=1.0)  # alpha, regularization parametresi
ridge_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_ridge = ridge_model.predict(X_test)

# Performans metrikleri
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = mean_squared_error(y_test, y_pred_ridge, squared=False)

print(f"Ridge Regression - R²: {r2_ridge}, MAE: {mae_ridge}, RMSE: {rmse_ridge}")
