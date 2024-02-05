import streamlit as st
from joblib import load
import pandas as pd

best_dt_model = load('best_decision_tree_model.joblib')
best_rf_model = load('best_random_forest_model.joblib')

X = load('X.joblib')

# Streamlit UI
def main():
    st.title('Dementia Prediction')
    st.write("A model to predict dementia in people :)")

    # Get user input
    diabetic = st.slider("Diabetic (1 for yes, 0 for no):", 0, 1, 0)
    body_temperature = st.slider("Body Temperature:", 35.0, 40.0, 36.8)
    blood_oxygen_level = st.slider("Blood Oxygen Level:", 90.0, 100.0, 95.5)
    weight = st.slider("Weight:", 50, 150, 70)
    mri_delay = st.slider("MRI Delay (in minutes):", 0, 60, 25)
    heart_rate = st.slider("Heart Rate:", 60, 100, 75)
    alcohol_level = st.slider("Alcohol Level:", 0.0, 1.0, 0.12)

    # Make predictions when a button is clicked
    if st.button('Make Predictions'):
        # Create a dictionary with user input
        user_input = {
            'Diabetic': diabetic,
            'BodyTemperature': body_temperature,
            'BloodOxygenLevel': blood_oxygen_level,
            'Weight': weight,
            'MRI_Delay': mri_delay,
            'HeartRate': heart_rate,
            'AlcoholLevel': alcohol_level
        }

        # Convert user input to a DataFrame
        user_df = pd.DataFrame([user_input])

        # Ensure columns in user_df are in the same order as X
        user_df = user_df[X.columns]

        # Make predictions
        dt_prediction = best_dt_model.predict(user_df)[0]
        rf_prediction = best_rf_model.predict(user_df)[0]

        # Display predictions
        st.write(f"Prediction according to Decision Tree: {dt_prediction}")
        st.write(f"Prediction according to Random Forest: {rf_prediction}")
        st.write("Note: 1 and 0 indicate the likelihood of dementia, 1 being yes, there's chances of having dementia and 0 being there's no chances of having dementia.")

if __name__ == "__main__":
    main()
