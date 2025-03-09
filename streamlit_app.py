"""
Weather Prediction Using Machine Learning Algorithms
=====================================================
This script processes weather data, performs data cleaning, visualizes trends, 
and implements machine learning models for weather prediction.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from windrose import WindroseAxes

# 1. Streamlit Application Title
st.title('Weather Prediction Using Machine Learning Algorithms')

# 2. Load the Dataset
st.subheader("Step 1: Load the Dataset")
df = pd.read_csv('https://raw.githubusercontent.com/SanketPaithankar/wheaterPredit/refs/heads/main/rajbhavan_combined.csv')
st.write("First 5 rows of the dataset:", df.head())

# 3. Handle Missing Values
st.subheader("Step 2: Handling Missing Values")
threshold = len(df) * 0.5
st.write("Dropping columns with more than 50% missing values")
df = df.dropna(thresh=threshold, axis=1)

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())
st.write("Missing values handled successfully")

# 4. Remove Duplicates
st.subheader("Step 3: Removing Duplicate Rows")
df = df.drop_duplicates()
st.write("Duplicates removed")

# 5. Handle Outliers using IQR
st.subheader("Step 4: Handling Outliers Using IQR")
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
Q1 = df[numerical_columns].quantile(0.25, numeric_only=True)
Q3 = df[numerical_columns].quantile(0.75, numeric_only=True)
IQR = Q3 - Q1
df = df.loc[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
st.write("Outliers removed using IQR method")

# 6. Normalize Data
st.subheader("Step 5: Normalizing Numerical Data")
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
st.write("Data normalization complete")

# 7. Save Cleaned Data
st.subheader("Step 6: Save Cleaned Data")
df.to_csv('rajbhavan_combined_cleaned_data.csv', index=False)
st.write("Cleaned dataset saved successfully")

# 8. Plot Temperature Trends
st.subheader("Step 7: Temperature Trends Over Time")
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Temp -  C'], label='Temp - C')
plt.plot(df['Date & Time'], df['High Temp -  C'], label='High Temp - C')
plt.plot(df['Date & Time'], df['Low Temp -  C'], label='Low Temp - C')
plt.xlabel('Date & Time')
plt.ylabel('Temperature (C)')
plt.title('Temperature Trends Over Time')
plt.legend()
st.pyplot(plt)

# 9. Plot Humidity Trends
st.subheader("Step 8: Humidity Trends Over Time")
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Hum - %'], label='Hum - %')
plt.plot(df['Date & Time'], df['Inside Hum - %'], label='Inside Hum - %')
plt.xlabel('Date & Time')
plt.ylabel('Humidity (%)')
plt.title('Humidity Trends Over Time')
plt.legend()
st.pyplot(plt)

# 10. Plot Rainfall Trends
st.subheader("Step 9: Rainfall Trends Over Time")
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Rain - in'], label='Rain - in')
plt.xlabel('Date & Time')
plt.ylabel('Rainfall (in)')
plt.title('Rainfall Trends Over Time')
plt.legend()
st.pyplot(plt)

# 11. Wind Rose Plot
st.subheader("Step 10: Wind Rose Plot")
wind_direction_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135,
    'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
    'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}
df['Wind_Direction_Degrees'] = df['Prevailing Wind Direction'].map(wind_direction_map)
df = df.dropna(subset=['Wind_Direction_Degrees', 'Avg Wind Speed - km/h'])
fig = plt.figure(figsize=(10, 10))
ax = WindroseAxes.from_ax(fig=fig)
ax.bar(df['Wind_Direction_Degrees'], df['Avg Wind Speed - km/h'], normed=False, opening=0.8, edgecolor='white')
ax.set_legend()
plt.title('Wind Rose Plot - Frequency of Wind Direction')
st.pyplot(fig)

st.write("Weather Prediction Data Processing and Visualization Completed Successfully!")
