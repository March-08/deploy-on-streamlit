# streamlit_app.py
import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris  # Load the same dataset used for training

# Load the Iris dataset for demonstration
iris = load_iris()
X, y = iris.data, iris.target

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app
st.title('Machine Learning Streamlit App')

# Sidebar for user input
st.sidebar.header('User Input')
user_input = []
for i in range(4):  # Assuming four features for simplicity
    user_input.append(st.sidebar.slider(f'Feature {i + 1}', float(np.min(X[:, i])), float(np.max(X[:, i])), float(np.mean(X[:, i]))))

# Make predictions
if st.button('Predict'):
    user_input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input_array)
    st.write(f'The model predicts class: {prediction[0]}')
