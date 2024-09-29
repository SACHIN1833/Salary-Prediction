import streamlit as st
import numpy as np
import pickle

# Function to load the model and encoders
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load model and encoders
data = load_model()
regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Display the salary prediction page
def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    # Available options for countries and education levels
    countries = (
        "United States", "India", "United Kingdom", "Germany", "Canada",
        "Brazil", "France", "Spain", "Australia", "Netherlands",
        "Poland", "Italy", "Russian Federation", "Sweden", "Other countries",
    )

    education_levels = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    # Input widgets
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)
    experience = st.slider("Years of Experience", 0, 50, 3)

    # Button to calculate salary
    ok = st.button("Calculate Salary")

    if ok:
        try:
            # Prepare input for the model
            X = np.array([[country, education, experience]])
            X[:, 0] = le_country.transform(X[:, 0])  # Transform country
            X[:, 1] = le_education.transform(X[:, 1])  # Transform education
            X = X.astype(float)  # Ensure input is float for the regressor

            # Make prediction
            salary = regressor.predict(X)
            st.subheader(f"The estimated salary is $ {salary[0]:,.2f}")
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Show the prediction page
show_predict_page()
