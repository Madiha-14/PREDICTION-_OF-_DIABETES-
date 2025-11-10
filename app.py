from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for model and preprocessing
model = None
model_threshold = 0.5
label_encoders = {}
feature_columns = []
training_columns = []
categorical_columns: list[str] = []
categorical_values_by_col: dict[str, list[str]] = {}
model_feature_names: list[str] = []

def load_model_and_setup():
    """Load the model and set up preprocessing"""
    global model, label_encoders, feature_columns
    
    try:
        # Try to load the final model first, then fallback to other models
        model_files = ['diabetes_model_final.pkl', 'diabetes_model_corrected.pkl', 'diabetes_model_optimized.pkl']
        model_loaded = False
        
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Handle new model format (dictionary with model info)
                if isinstance(model_data, dict):
                    model = model_data['model']
                    globals()['model_feature_names'] = model_data.get('feature_names', [])
                    globals()['model_threshold'] = model_data.get('threshold', 0.5)
                    print(f"✅ New model loaded from {model_file}!")
                    print(f"📊 Model info: Accuracy={model_data.get('accuracy', 'N/A'):.4f}")
                    print(f"🎯 Threshold: {globals()['model_threshold']:.3f}")
                else:
                    # Handle old model format (just the model object)
                    model = model_data
                    globals()['model_threshold'] = 0.5
                    print(f"✅ Old model loaded from {model_file}!")
                
                model_loaded = True
                break
            except FileNotFoundError:
                print(f"⚠️ Model file {model_file} not found")
                continue
        
        if not model_loaded:
            print("❌ No model file found!")
            return False
        
        # Load and analyze the training data to understand the expected format
        df = pd.read_csv('diabetes_corrected.csv')

        # Create binary target (since your complex models had issues)
        df['Has_Diabetes'] = df['Diabetes_Type'].notna().astype(int)

        # Prepare base features
        # Do not drop 'Prediabetes' because the trained model may include its one-hot columns
        X_base = df.drop(['ID', 'Diabetes_Type', 'Has_Diabetes'], axis=1, errors='ignore')

        # Record categorical columns and their observed categories
        nonlocal_categorical_columns = list(X_base.select_dtypes(include=['object']).columns)
        globals()['categorical_columns'] = nonlocal_categorical_columns
        globals()['categorical_values_by_col'] = {}

        X_for_dummies = X_base.copy()
        for col in nonlocal_categorical_columns:
            # Fill missing categories with explicit 'None'
            X_for_dummies[col] = X_for_dummies[col].fillna('None')
            observed_values = list(pd.Series(X_for_dummies[col].unique()).dropna().astype(str))
            # Ensure 'None' exists as a category if present in data or add as safe default
            if 'None' not in observed_values:
                observed_values.append('None')
            categorical_values_by_col[col] = observed_values

        # One-hot encode to derive training column names to match the trained model
        X_dummies = pd.get_dummies(X_for_dummies, columns=nonlocal_categorical_columns, drop_first=False)
        globals()['training_columns'] = list(X_dummies.columns)

        # If the loaded model carries feature names, prefer those to ensure exact alignment
        if hasattr(model, 'feature_names_in_'):
            globals()['model_feature_names'] = list(getattr(model, 'feature_names_in_'))
            globals()['training_columns'] = globals()['model_feature_names']
        else:
            globals()['model_feature_names'] = globals()['training_columns']

        # Backward compatibility for any prior logic using feature_columns
        globals()['feature_columns'] = list(globals()['training_columns'])

        print(f"✅ Preprocessing setup complete. Features: {len(training_columns)}")
        print(f"📋 Categorical features: {list(nonlocal_categorical_columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_input(data):
    """Preprocess input data to match training format"""
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Ensure all expected raw feature columns exist
        # Add a default for Prediabetes since the model may expect its one-hot columns
        if 'Prediabetes' not in input_df.columns:
            input_df['Prediabetes'] = 'No'

        for col in categorical_columns:
            if col not in input_df.columns:
                # Fill missing categorical with safe default 'None' or first observed value
                default_val = 'None' if 'None' in categorical_values_by_col.get(col, []) else (categorical_values_by_col.get(col, [''])[0] if categorical_values_by_col.get(col) else 'None')
                input_df[col] = default_val
        
        # Type coercion for numerics: any non-categorical columns will be treated as numeric
        numeric_like_cols = [c for c in input_df.columns if c not in categorical_columns]
        for c in numeric_like_cols:
            try:
                input_df[c] = pd.to_numeric(input_df[c])
            except Exception:
                pass

        # Fill NaNs in categorical columns
        for col in categorical_columns:
            input_df[col] = input_df[col].fillna('None').astype(str)
            # Map unknown categories to a safe default
            allowed = set(categorical_values_by_col.get(col, []))
            if len(input_df[col]) > 0 and input_df[col].iloc[0] not in allowed and len(allowed) > 0:
                input_df[col] = allowed.__iter__().__next__()

        # One-hot encode input to match training representation
        input_for_dummies = input_df.copy()
        input_for_dummies = pd.get_dummies(input_for_dummies, columns=categorical_columns, drop_first=False)

        # Align to the model's expected feature names exactly
        expected_cols = model_feature_names if model_feature_names else training_columns
        input_for_dummies = input_for_dummies.reindex(columns=expected_cols, fill_value=0)

        return input_for_dummies
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise e

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Diabetes Prediction API is running',
        'model_loaded': model is not None,
        'features_count': len(feature_columns) if feature_columns else 0,
        'endpoints': ['/predict', '/']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get JSON data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        print(f"📨 Received prediction request: {data}")
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        print(f"🔄 Processed data shape: {processed_data.shape}")
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction_prob = model.predict_proba(processed_data)[0]
            
            # Use the threshold from the loaded model
            threshold = model_threshold
            
            # Apply threshold to get prediction
            diabetic_probability = float(prediction_prob[1]) if len(prediction_prob) > 1 else float(prediction_prob[0])
            prediction = 1 if diabetic_probability >= threshold else 0
            
            # Convert to readable format
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
            confidence = {
                'Non-Diabetic': float(prediction_prob[0]) if len(prediction_prob) > 1 else float(1 - prediction_prob[0]),
                'Diabetic': diabetic_probability
            }
        else:
            # Fallback for models without predict_proba
            prediction = model.predict(processed_data)[0]
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
            confidence = {'prediction_score': float(prediction)}
        
        response = {
            'prediction': result,
            'confidence': confidence,
            'raw_prediction': int(prediction) if hasattr(prediction, '__int__') else str(prediction)
        }
        
        print(f"✅ Prediction successful: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'available_endpoints': ['/', '/predict']}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction API Server...")
    
    # Load model and setup preprocessing
    if load_model_and_setup():
        print("✅ Server setup complete!")
        print("🌐 Server will be available at: http://127.0.0.1:5000")
        print("🩺 Helath Check at Full website @ ")
        
        # Start the Flask server
        app.run(host='127.0.0.1', port=5000, debug=True)
    else:
        print("❌ Failed to start server due to model loading issues")
