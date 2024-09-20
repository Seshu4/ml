import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit app title
st.title('Bank Nifty Data Analysis and Linear Regression Model')

# Upload CSV file via Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# Check if file is uploaded
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Preprocess the data
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

    # Clean the data by replacing '-' with NaN and dropping rows with NaN values
    data_cleaned = data.replace('-', pd.NA).dropna()

    # Convert columns to numeric where applicable
    data_cleaned[['Open', 'High', 'Low', 'Close', 'Volume']] = data_cleaned[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # Show dataset in Streamlit
    st.write('Data Preview:')
    st.dataframe(data_cleaned.head())

    # Plot the closing price over time
    st.subheader('Bank Nifty Closing Price Over Time')
    plt.figure(figsize=(10, 6))
    plt.plot(data_cleaned['Date'], data_cleaned['Close'], label='Close Price')
    plt.title('Bank Nifty Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(plt)

    # Calculate the correlation matrix
    correlation_matrix = data_cleaned[['Open', 'High', 'Low', 'Volume', 'Close']].corr()

    # Show the correlation matrix in Streamlit
    st.subheader('Correlation Matrix:')
    st.write(correlation_matrix)

    # Visualize the correlation matrix with a heatmap
    st.subheader('Correlation Matrix Heatmap')
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)

    # Visualize the relationship between each independent variable and the target variable (Close Price)
    st.subheader('Scatter Plots of Independent Variables vs Close Price')
    for column in ['Open', 'High', 'Low', 'Volume']:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data_cleaned[column], y=data_cleaned['Close'])
        plt.title(f'{column} vs Close Price')
        plt.xlabel(column)
        plt.ylabel('Close Price')
        st.pyplot(plt)

    # Prepare the data for machine learning
    X = data_cleaned[['Open', 'High', 'Low', 'Volume']]  # Features
    y = data_cleaned['Close']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the model's performance
    mse = mean_squared_error(y_test, y_pred)

    # Show the performance metrics and model coefficients in Streamlit
    st.subheader('Model Performance')
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Model Coefficients: {model.coef_}")
    st.write(f"Intercept: {model.intercept_}")
else:
    st.write("Please upload a CSV file to proceed.")
