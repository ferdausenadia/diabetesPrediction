import gradio as gr
import pandas as pd
import numpy as np
import pickle


# Load trained model

with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)


# Prediction function

def predict_diabetes(pregnancies, glucose, bloodpressure,
                     insulin, bmi, diabetespedigree, age):

    # DataFrame 
    input_df = pd.DataFrame([[pregnancies, glucose, bloodpressure,
                              insulin, bmi,
                              diabetespedigree, age]],
                            columns=[
                                'Pregnancies',
                                'Glucose',
                                'BloodPressure',
                                'Insulin',
                                'BMI',
                                'DiabetesPedigreeFunction',
                                'Age'
                            ])

    # Predict
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 1:
        return "Diabetic"
    else:
        return " Not Diabetic"


inputs = [
    gr.Number(label="Pregnancies"),
    gr.Number(label="Glucose"),
    gr.Number(label="Blood Pressure"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="Diabetes Pedigree Function"),
    gr.Number(label="Age")
]


app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction System",
    description="Enter patient details to predict diabetes risk "
)

# Launch app
app.launch(share=True)