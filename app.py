import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Saved Model and Scaler ---

# Load the trained model
try:
    model = joblib.load('simple_model.pkl')
except FileNotFoundError:
    st.error("Error: 'simple_model.pkl' not found. Make sure the file is in the same directory.")
    st.stop()

# Load the scaler
try:
    scaler = joblib.load('simple_scaler.pkl')
except FileNotFoundError:
    st.error("Error: 'simple_scaler.pkl' not found. Make sure the file is in the same directory.")
    st.stop()

# --- 2. Set Up the Web App Interface ---

st.set_page_config(page_title="Salary Predictor", page_icon="💼")
st.title('💼 Employee Salary Predictor')
st.write("This app predicts whether an employee's income is more or less than $50,000 based on their details.")

st.sidebar.header('User Input Features')

# Create input fields in the sidebar
age = st.sidebar.slider('Age', 17, 90, 35)
educational_num = st.sidebar.slider('Education Level (Numeric)', 1, 16, 10)
capital_gain = st.sidebar.number_input('Capital Gain', 0, 99999, 0)
capital_loss = st.sidebar.number_input('Capital Loss', 0, 4356, 0)
hours_per_week = st.sidebar.slider('Hours per Week', 1, 99, 40)

# --- 3. Prediction Logic ---

# Create a button to make a prediction
if st.sidebar.button('Predict Salary'):
    # Create a DataFrame from the user's input
    # The order MUST match the order used to train the model
    input_data = pd.DataFrame(
        [[age, educational_num, capital_gain, capital_loss, hours_per_week]],
        columns=['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    )
    
    # Scale the user's input data
    input_scaled = scaler.transform(input_data)
    
    # Make the prediction (model outputs 0 or 1)
    prediction = model.predict(input_scaled)
    
    # Get the probability of the prediction
    prediction_proba = model.predict_proba(input_scaled)

    # --- 4. Display the Result ---
    
    st.subheader('Prediction Result')
    
    if prediction[0] == 1:
        st.success('**Income is likely > $50K**')
        st.write(f"Confidence: {prediction_proba[0][1] * 100:.2f}%")
    else:
        st.error('**Income is likely <= $50K**')
        st.write(f"Confidence: {prediction_proba[0][0] * 100:.2f}%")

    st.write("---")
    st.subheader("User Input Details:")
    st.write(input_data)

else:
    st.info('Please enter details in the sidebar and click "Predict Salary".')