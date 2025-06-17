"""
Flask Web Application for Mushroom Classification
Interactive demo using the trained CatBoost model
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import json
import os
from typing import Dict, Any, List

app = Flask(__name__)

# Global variables
model = None
feature_mappings = {}
feature_order = []

def load_model_and_mappings():
    """Load the trained model and feature mappings"""
    global model, feature_mappings, feature_order
    
    # Get the directory of the current file (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(current_dir)
    
    # Create dynamic paths
    model_path = os.path.join(project_root, 'models', 'catboost_model.cbm')
    mappings_path = os.path.join(current_dir, 'utils', 'feature_mappings.json')
    
    # Load the CatBoost model
    if os.path.exists(model_path):
        model = CatBoostClassifier()
        model.load_model(model_path)
        feature_order = model.feature_names_
        if feature_order:
            print(f"Model loaded successfully with {len(feature_order)} features")
        else:
            print("Model loaded but feature names not available")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load feature mappings
    if os.path.exists(mappings_path):
        with open(mappings_path, 'r') as f:
            feature_mappings = json.load(f)
        print("Feature mappings loaded successfully")
    else:
        raise FileNotFoundError(f"Feature mappings not found at {mappings_path}")
    
    return model  # Return the model to ensure it's properly loaded

def prepare_input_data(form_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert form data to model input format"""
    # Create a dictionary with all features
    input_dict = {}
    
    # Ensure feature_order is available
    if not feature_order:
        raise ValueError("Feature order not available. Model may not be loaded properly.")
    
    for feature in feature_order:
        value = form_data.get(feature, '')
        
        # Handle numeric features
        if feature in ['cap-diameter', 'stem-height', 'stem-width']:
            try:
                input_dict[feature] = float(value) if value else 0.0
            except ValueError:
                input_dict[feature] = 0.0
        else:
            # Handle categorical features
            input_dict[feature] = value if value else 'unknown'
    
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    return df

@app.route('/')
def index():
    """Main page with the mushroom classification form"""
    return render_template('index.html', 
                         feature_mappings=feature_mappings,
                         feature_order=feature_order)

def generate_image_prompt(form_data: Dict[str, Any], prediction: str) -> str:
    """Generate a detailed ChatGPT prompt for mushroom image generation"""
    
    # Extract key visual features
    cap_shape = feature_mappings.get('cap-shape', {}).get(form_data.get('cap-shape', ''), 'unknown')
    cap_color = feature_mappings.get('cap-color', {}).get(form_data.get('cap-color', ''), 'unknown')
    cap_surface = feature_mappings.get('cap-surface', {}).get(form_data.get('cap-surface', ''), 'unknown')
    stem_color = feature_mappings.get('stem-color', {}).get(form_data.get('stem-color', ''), 'unknown')
    gill_color = feature_mappings.get('gill-color', {}).get(form_data.get('gill-color', ''), 'unknown')
    habitat = feature_mappings.get('habitat', {}).get(form_data.get('habitat', ''), 'unknown')
    has_ring = feature_mappings.get('has-ring', {}).get(form_data.get('has-ring', ''), 'unknown')
    
    # Get dimensions
    cap_diameter = form_data.get('cap-diameter', 'medium-sized')
    stem_height = form_data.get('stem-height', 'medium')
    stem_width = form_data.get('stem-width', 'medium')
    
    # Create descriptive prompt
    prompt = f"""Create a detailed, photorealistic image of a mushroom with the following characteristics:

**Cap Features:**
- Shape: {cap_shape}
- Color: {cap_color}
- Surface texture: {cap_surface}
- Diameter: {cap_diameter} cm

**Stem Features:**
- Color: {stem_color}
- Height: {stem_height} cm
- Width: {stem_width} cm
- Ring: {'Has a ring around the stem' if has_ring == 'true' else 'No ring on stem'}

**Gills:**
- Color: {gill_color}

**Environment:**
- Natural habitat: {habitat}
- Show the mushroom in its natural environment

**Style Requirements:**
- Photorealistic, high-quality nature photography style
- Natural lighting, forest atmosphere
- Sharp focus on the mushroom
- Beautiful bokeh background
- Professional nature photography composition

**Safety Note:** This is a {'potentially edible' if prediction == 'Edible' else 'poisonous'} mushroom species based on the classification."""

    return prompt

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Ensure model is loaded
        if model is None:
            raise ValueError("Model not loaded. Please restart the application.")
        
        # Get form data
        form_data = request.form.to_dict()
        
        # Prepare input data
        input_df = prepare_input_data(form_data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Calculate confidence
        confidence = max(probability) * 100
        
        # Determine result
        result = "Edible" if prediction == 0 else "Poisonous"
        edible_prob = probability[0] * 100
        poisonous_prob = probability[1] * 100
        
        # Determine confidence level
        if confidence > 90:
            confidence_level = "Very High"
        elif confidence > 80:
            confidence_level = "High"
        elif confidence > 70:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Generate image prompt
        image_prompt = generate_image_prompt(form_data, result)
        
        response = {
            'success': True,
            'prediction': result,
            'edible_probability': round(edible_prob, 2),
            'poisonous_probability': round(poisonous_prob, 2),
            'confidence': round(confidence, 2),
            'confidence_level': confidence_level,
            'image_prompt': image_prompt,
            'warning': 'NEVER eat wild mushrooms based on this prediction alone!' if result == "Edible" else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/features')
def get_features():
    """API endpoint to get feature information"""
    return jsonify({
        'features': feature_order,
        'mappings': feature_mappings
    })

if __name__ == '__main__':
    import socket
    
    def is_port_available(port):
        """Check if a port is available for use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except socket.error:
            return False
    
    def find_available_port(start_port=5000, max_attempts=10):
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            if is_port_available(port):
                return port
        return None
    
    try:
        load_model_and_mappings()
        print("Starting Mushroom Classification Web App...")
        
        # Try to find an available port
        port = find_available_port(5000)
        if port is None:
            print("Error: Could not find an available port between 5000-5009")
            exit(1)
        
        print(f"Visit http://localhost:{port} to use the application")
        app.run(debug=True, host='0.0.0.0', port=port)
        
    except Exception as e:
        print(f"Error starting application: {e}")
