import json
import numpy as np
import gradio as gr
import pickle
from sklearn.linear_model import LinearRegression

# Load the model
with open('HYD_Rent_Predictor.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)
    if isinstance(best_model, LinearRegression):
        best_model.check_input = False

# Load the columns
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

def predict_price(locality, balconies, bathroom, furnishingDesc, parking, property_size, type_bhk, floor):
    loc_index = np.where(np.array(data_columns) == locality.lower())[0][0]

    x = np.zeros(len(data_columns))
    x[0] = balconies
    x[1] = bathroom
    x[2] = furnishingDesc
    x[3] = parking
    x[4] = property_size
    x[5] = type_bhk
    x[6] = floor

    if loc_index >= 0:
        x[loc_index] = 1

    return best_model.predict([x])[0]

# Gradio interface
def interface(locality, balconies, bathroom, furnishingDesc, parking, property_size, type_bhk, floor):
    result = predict_price(locality, balconies, bathroom, furnishingDesc, parking, property_size, type_bhk, floor)
    return f"Predicted Rent: {result:.2f} INR"

furnishing_options = [0.5, 0, 1]  # Replace with actual options
parking_options = [0, 1, 2, 3]  # Replace with actual options
type_bhk_options = [0.5, 1, 2, 3, 4, 5]  # Replace with actual options

inputs = [
    gr.Textbox(label="Locality"),
    gr.Slider(minimum=0, maximum=7, value=1, step=1, label="Balconies"),
    gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Bathrooms"),
    gr.Dropdown(furnishing_options, label="Furnishing Description"),
    gr.Dropdown(parking_options, label="Parking"),
    gr.Slider(minimum=100, maximum=13000, value=1000, step=150, label="Property Size (in sqft)"),
    gr.Dropdown(type_bhk_options, label="Type BHK"),
    gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Floor"),
]

outputs = gr.Textbox()

# Create Gradio interface
gr.Interface(fn=interface, inputs=inputs, outputs=outputs, title="Hyderabad House Rent Prediction").launch(share=True)