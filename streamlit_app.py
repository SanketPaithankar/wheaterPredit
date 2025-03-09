import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from windrose import WindroseAxes

# Streamlit app title
st.title('Weather Prediction Using Machine Learning Algorithms')

# Read the CSV file
df = pd.read_csv('https://raw.githubusercontent.com/SanketPaithankar/wheaterPredit/refs/heads/main/rajbhavan_combined.csv')

# Display the first few rows of the dataframe
st.write("### Preview of Dataset", df.head())

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

# Save cleaned data
df.to_csv('rajbhavan_combined_cleaned_data.csv', index=False)

# Display column names
st.write("### Column Names", df.columns)

# Temperature Trends
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Temp -  C'], label='Temperature (°C)', color='red')
plt.plot(df['Date & Time'], df['High Temp -  C'], label='High Temperature (°C)', color='orange')
plt.plot(df['Date & Time'], df['Low Temp -  C'], label='Low Temperature (°C)', color='blue')
plt.xlabel('Date & Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trends Over Time')
plt.legend()
st.pyplot(plt)

# Humidity Trends
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Hum - %'], label='Humidity (%)', color='green')
plt.plot(df['Date & Time'], df['Inside Hum - %'], label='Inside Humidity (%)', color='purple')
plt.xlabel('Date & Time')
plt.ylabel('Humidity (%)')
plt.title('Humidity Trends Over Time')
plt.legend()
st.pyplot(plt)

# Rainfall Trends
plt.figure(figsize=(14, 7))
plt.plot(df['Date & Time'], df['Rain - in'], label='Rainfall (inches)', color='blue')
plt.xlabel('Date & Time')
plt.ylabel('Rainfall (inches)')
plt.title('Rainfall Trends Over Time')
plt.legend()
st.pyplot(plt)

# Wind Rose Plot
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
