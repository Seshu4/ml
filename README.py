import streamlit as st

# Title of the app
st.title("Hello, World Streamlit App")

# Subtitle or header
st.header("Welcome to your first Streamlit Application")

# Display some text
st.write("This is a simple Streamlit app to get you started!")

# Adding a button
if st.button("Click me"):
    st.write("Hello, Streamlit user!")

# Text input from the user
user_input = st.text_input("Enter your name:", "")

# Displaying the user input
if user_input:
    st.write(f"Hello, {user_input}! Thanks for trying out Streamlit.")

# Checkbox example
if st.checkbox("Show more details"):
    st.write("Streamlit makes it easy to build web apps with Python!")

# Number input
number = st.slider("Pick a number", 0, 100)
st.write(f"Your chosen number is: {number}")
