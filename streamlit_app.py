"""
Weather Prediction Using Machine Learning Algorithms
=====================================================
This script processes weather data, performs data cleaning, visualizes trends, 
and implements machine learning models for weather prediction.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Streamlit App Title
st.title('Weather Prediction Using Machine Learning Algorithms')

# Load Dataset
DATA_URL = 'https://raw.githubusercontent.com/SanketPaithankar/wheaterPredit/refs/heads/main/rajbhavan_combined.csv'
df = pd.read_csv(DATA_URL)

# Display first few rows (Before Cleaning)
st.subheader("Raw Data (Before Cleaning)")
st.write(df.head())

# Data Preprocessing
## Convert Date & Time to datetime format
df['Date & Time'] = pd.to_datetime(df['Date & Time'], errors='coerce')

## Handling Missing Values
threshold = len(df) * 0.5  # Drop columns with more than 50% missing values
df = df.dropna(thresh=threshold, axis=1)

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())

# Drop duplicate entries
df = df.drop_duplicates()

# Convert numeric columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values after conversion
df = df.dropna()

# Normalization
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display cleaned dataset
st.subheader("Cleaned Data (After Preprocessing)")
st.write(df.head())

# Data Visualization
st.subheader("Data Visualizations")

# Temperature Trends
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Temp - C'], label='Temp - C', alpha=0.7)
plt.plot(df['Date & Time'], df['High Temp - C'], label='High Temp - C', alpha=0.7)
plt.plot(df['Date & Time'], df['Low Temp - C'], label='Low Temp - C', alpha=0.7)
plt.xlabel('Date & Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trends Over Time')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# Humidity Trends
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Hum - %'], label='Humidity - %', alpha=0.7)
plt.plot(df['Date & Time'], df['Inside Hum - %'], label='Inside Hum - %', alpha=0.7)
plt.xlabel('Date & Time')
plt.ylabel('Humidity (%)')
plt.title('Humidity Trends Over Time')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# Rainfall Trends
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Rain - in'], label='Rainfall (in)', alpha=0.7)
plt.xlabel('Date & Time')
plt.ylabel('Rainfall (in)')
plt.title('Rainfall Trends Over Time')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# Wind Rose Plot
st.subheader("Wind Rose Plot")
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

st.write("Dataset successfully cleaned, visualized, and ready for modeling!")
