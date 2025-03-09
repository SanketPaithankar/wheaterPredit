import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st

# Make Predictions
# Use the selected model to make predictions based on user input and display the results on the screen.

# Streamlit app
st.title('Weather Prediction')

df = pd.read_csv('https://raw.githubusercontent.com/SanketPaithankar/wheaterPredit/refs/heads/main/rajbhavan_combined.csv')
