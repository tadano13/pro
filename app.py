import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained Linear Regression model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature columns (updated)
feature_columns = ['Age', 'Gender', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 
                   'Extracurricular', 'Sports', 'Music', 'Volunteering']

# Define options for categorical features (updated)
categorical_options = {
    'Gender': ['Male', 'Female'],
    'Tutoring': ['Yes', 'No'],
    'ParentalSupport': ['High', 'Medium', 'Low'],
    'Extracurricular': ['Yes', 'No'],
    'Sports': ['Yes', 'No'],
    'Music': ['Yes', 'No'],
    'Volunteering': ['Yes', 'No']
}

# Create a Streamlit app
st.title('Student CGPA Prediction')
st.text('made by Nishat Doma Sawaimoon')

# Create input fields for the user
inputs = {}
for feature in feature_columns:
    if feature in ['Age', 'StudyTimeWeekly', 'Absences']:  # Numerical features
        inputs[feature] = st.slider(f'{feature}', min_value=0, max_value=100, value=0)  # Adjust max_value as needed
    elif feature in categorical_options:  # Categorical features
        inputs[feature] = st.selectbox(f'{feature}', options=categorical_options[feature])

# Button to make a prediction
if st.button('Predict CGPA'):
    # Convert inputs to a DataFrame
    input_df = pd.DataFrame([inputs], columns=feature_columns)
    
    # Encode categorical features as needed
    le = LabelEncoder()
    for column in input_df.select_dtypes(include=['object']).columns:
        if column in categorical_options:  # Ensure encoding matches training
            le.fit(categorical_options[column])
            input_df[column] = le.transform(input_df[column])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display the result
    st.write(f'Predicted CGPA: {prediction[0]:.2f}')
