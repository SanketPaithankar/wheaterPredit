import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Streamlit app title
st.title('Weather Prediction Using Machine Learning Algorithms')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/SanketPaithankar/wheaterPredit/refs/heads/main/rajbhavan_combined.csv')

# Display raw data
st.subheader("Sample Raw Data")
st.dataframe(df.head())

# Drop columns with more than 50% missing values
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())

# Drop duplicates
df = df.drop_duplicates()

# Handle outliers using IQR
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
Q1 = df[numerical_columns].quantile(0.25, numeric_only=True)
Q3 = df[numerical_columns].quantile(0.75, numeric_only=True)
IQR = Q3 - Q1
df = df.loc[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | 
              (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalize numerical columns
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display cleaned data
st.subheader("Cleaned Data Preview")
st.dataframe(df.head())

df.to_csv('rajbhavan_combined_cleaned_data.csv', index=False)

# Plot temperature trends
st.subheader("Temperature Trends Over Time")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['Date & Time'], df['Temp -  C'], label='Temp - C')
ax.plot(df['Date & Time'], df['High Temp -  C'], label='High Temp - C')
ax.plot(df['Date & Time'], df['Low Temp -  C'], label='Low Temp - C')
ax.set_xlabel('Date & Time')
ax.set_ylabel('Temperature (C)')
ax.set_title('Temperature Trends Over Time')
ax.legend()
st.pyplot(fig)

# Plot humidity trends
st.subheader("Humidity Trends Over Time")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['Date & Time'], df['Hum - %'], label='Hum - %')
ax.plot(df['Date & Time'], df['Inside Hum - %'], label='Inside Hum - %')
ax.set_xlabel('Date & Time')
ax.set_ylabel('Humidity (%)')
ax.set_title('Humidity Trends Over Time')
ax.legend()
st.pyplot(fig)

# Plot rainfall trends
st.subheader("Rainfall Trends Over Time")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['Date & Time'], df['Rain - in'], label='Rain - in')
ax.set_xlabel('Date & Time')
ax.set_ylabel('Rainfall (in)')
ax.set_title('Rainfall Trends Over Time')
ax.legend()
st.pyplot(fig)

# Wind Rose Plot
st.subheader("Wind Rose Plot - Frequency of Wind Direction")
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
st.pyplot(fig)
