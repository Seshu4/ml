import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit app title
st.title('Bank Nifty Closing Price Prediction')

# Load the dataset
data = pd.read_csv(r'C:\Users\manohar\OneDrive\Documents\bank_nifty.csv')

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Clean the data by replacing '-' with NaN and dropping rows with NaN values
data_cleaned = data.replace('-', pd.NA).dropna()

# Convert columns to numeric where applicable
data_cleaned[['Open', 'High', 'Low', 'Close', 'Volume']] = data_cleaned[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

# Prepare the data for machine learning
X = data_cleaned[['Open', 'High', 'Low', 'Volume']]  # Features
y = data_cleaned['Close']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Input feature values for prediction
st.subheader('Enter the following details to predict Closing Price:')
open_input = st.number_input("Open Price", value=0.0)
high_input = st.number_input("High Price", value=0.0)
low_input = st.number_input("Low Price", value=0.0)
volume_input = st.number_input("Volume", value=0.0)

# Predict based on user input
if st.button('Predict Close Price'):
    user_data = pd.DataFrame([[open_input, high_input, low_input, volume_input]], 
                             columns=['Open', 'High', 'Low', 'Volume'])
    prediction = model.predict(user_data)
    st.write(f"Predicted Close Price: {prediction[0]}")
