# Weather Prediction Using Machine Learning Algorithms
# =====================================================
# This script processes weather data, performs data cleaning, visualizes trends, 
# and implements machine learning models for weather prediction.

# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from sklearn.preprocessing import MinMaxScaler

# Streamlit App Title
st.title('Weather Prediction Using Machine Learning Algorithms')

# Load Dataset
DATA_URL = 'https://raw.githubusercontent.com/SanketPaithankar/wheaterPredit/refs/heads/main/rajbhavan_combined.csv'
df = pd.read_csv(DATA_URL)

# Display first few rows (Raw Data)
st.subheader("Raw Dataset (Before Cleaning)")
st.write(df.head())

# Data Visualization (Before Cleaning)
st.subheader("Raw Data Visualizations")

# Convert Date & Time to datetime format (Handle errors)
if 'Date & Time' in df.columns:
    df['Date & Time'] = pd.to_datetime(df['Date & Time'], errors='coerce')

# Temperature Trends (Raw Data)
if 'Temp -  C' in df.columns and 'High Temp -  C' in df.columns and 'Low Temp -  C' in df.columns:
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date & Time'], df['Temp -  C'], label='Temp - C', alpha=0.7)
    plt.plot(df['Date & Time'], df['High Temp -  C'], label='High Temp - C', alpha=0.7)
    plt.plot(df['Date & Time'], df['Low Temp -  C'], label='Low Temp - C', alpha=0.7)
    plt.xlabel('Date & Time')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature Trends Over Time (Raw Data)')
    plt.legend()
    st.pyplot(plt)

# Humidity Trends (Raw Data)
if 'Hum - %' in df.columns:
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date & Time'], df['Hum - %'], label='Humidity - %', alpha=0.7)
    plt.xlabel('Date & Time')
    plt.ylabel('Humidity (%)')
    plt.title('Humidity Trends Over Time (Raw Data)')
    plt.legend()
    st.pyplot(plt)

# Rainfall Trends (Raw Data)
if 'Rain - in' in df.columns:
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date & Time'], df['Rain - in'], label='Rainfall (in)', alpha=0.7)
    plt.xlabel('Date & Time')
    plt.ylabel('Rainfall (in)')
    plt.title('Rainfall Trends Over Time (Raw Data)')
    plt.legend()
    st.pyplot(plt)

# Data Preprocessing
st.subheader("Data Cleaning and Preprocessing")

# Handling Missing Values
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

# Handling Outliers using IQR
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
Q1 = df[numerical_columns].quantile(0.25, numeric_only=True)
Q3 = df[numerical_columns].quantile(0.75, numeric_only=True)
IQR = Q3 - Q1
df = df.loc[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | 
              (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalization
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display cleaned dataset
st.subheader("Cleaned Dataset (After Processing)")
st.write(df.head())

# Data Visualization (After Cleaning)
st.subheader("Cleaned Data Visualizations")

# Temperature Trends (Cleaned Data)
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Temp -  C'], label='Temp - C', alpha=0.7)
plt.plot(df['Date & Time'], df['High Temp -  C'], label='High Temp - C', alpha=0.7)
plt.plot(df['Date & Time'], df['Low Temp -  C'], label='Low Temp - C', alpha=0.7)
plt.xlabel('Date & Time')
plt.ylabel('Temperature (C)')
plt.title('Temperature Trends Over Time (Cleaned Data)')
plt.legend()
st.pyplot(plt)

# Humidity Trends (Cleaned Data)
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Hum - %'], label='Humidity - %', alpha=0.7)
plt.xlabel('Date & Time')
plt.ylabel('Humidity (%)')
plt.title('Humidity Trends Over Time (Cleaned Data)')
plt.legend()
st.pyplot(plt)

# Rainfall Trends (Cleaned Data)
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Rain - in'], label='Rainfall (in)', alpha=0.7)
plt.xlabel('Date & Time')
plt.ylabel('Rainfall (in)')
plt.title('Rainfall Trends Over Time (Cleaned Data)')
plt.legend()
st.pyplot(plt)

# Wind Rose Plot
st.subheader("Wind Rose Plot")
if 'Prevailing Wind Direction' in df.columns and 'Avg Wind Speed - km/h' in df.columns:
    wind_direction_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135,
        'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
        'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    df['Wind_Direction_Degrees'] = df['Prevailing Wind Direction'].map(wind_direction_map)
    df = df.dropna(subset=['Wind_Direction_Degrees', 'Avg Wind Speed - km/h'])
    
    try:
        fig = plt.figure(figsize=(10, 10))
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(df['Wind_Direction_Degrees'], df['Avg Wind Speed - km/h'], normed=False, opening=0.8, edgecolor='white')
        ax.set_legend()
        plt.title('Wind Rose Plot - Frequency of Wind Direction')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Wind Rose Plot: {e}")

st.write("Dataset successfully cleaned, visualized, and ready for modeling!")
