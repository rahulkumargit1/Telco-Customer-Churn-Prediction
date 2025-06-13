from flask import Flask, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import numpy as np
import joblib
import logging
import io
from pathlib import Path
from churn_prediction import predict_new_customers

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_churn_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_churn_prediction")

# Load the trained model
model_path = Path('churn_prediction_results/churn_prediction_model.pkl')
try:
    model = joblib.load(model_path)
    logger.info(f"Successfully loaded model from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Setup rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Feature engineering function
def calculate_engineered_features(df):
    try:
        df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure'].clip(lower=1)
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
        df['ServiceCount'] = df[['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                'StreamingMovies']].eq('Yes').sum(axis=1)
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        df['ContractRiskFactor'] = df['Contract'].map(contract_risk)
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

# Single prediction endpoint
@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
        
        logger.info(f"Received data: {data}")
        
        # Convert input data to DataFrame
        df = pd.DataFrame(data)
        logger.info(f"DataFrame created with shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Ensure all required columns are present
        required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
        
        # Convert data types to match model expectations
        try:
            df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
            df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
            df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Check for NaN values after conversion
            invalid_cols = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']].isnull().any()
            if invalid_cols.any():
                invalid_cols = invalid_cols[invalid_cols].index.tolist()
                logger.error(f"Data type conversion resulted in NaN values in columns: {invalid_cols}")
                return jsonify({'error': f'Invalid numeric values in columns: {", ".join(invalid_cols)}'}), 400
        except Exception as e:
            logger.error(f"Data type conversion error: {str(e)}")
            return jsonify({'error': f'Data type conversion error: {str(e)}'}), 400
        
        # Validate numeric inputs
        if (df['tenure'] < 0).any() or (df['tenure'] > 120).any():
            logger.error("Tenure validation failed")
            return jsonify({'error': 'Tenure must be between 0 and 120 months'}), 400
        if (df['MonthlyCharges'] <= 0).any() or (df['MonthlyCharges'] > 500).any():
            logger.error("MonthlyCharges validation failed")
            return jsonify({'error': 'MonthlyCharges must be between 0 and 500'}), 400
        if (df['TotalCharges'] <= 0).any() or (df['TotalCharges'] > 10000).any():
            logger.error("TotalCharges validation failed")
            return jsonify({'error': 'TotalCharges must be between 0 and 10000'}), 400
        
        # Apply feature engineering
        df = calculate_engineered_features(df)
        
        # Validate engineered features
        if df[['AvgMonthlyCharges', 'CLV', 'ServiceCount', 'ContractRiskFactor']].isnull().any().any():
            logger.error("Computed features contain NaN values")
            return jsonify({'error': 'Computed features (AvgMonthlyCharges, CLV, ServiceCount, or ContractRiskFactor) are invalid'}), 400
        
        logger.info(f"Data after preprocessing:\n{df.dtypes}")
        
        # Make predictions
        try:
            results = predict_new_customers(model, df)
        except Exception as e:
            logger.error(f"Error in predict_new_customers: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
        
        # Convert results to JSON
        results_json = results.to_dict(orient='records')
        logger.info(f"Predictions made successfully: {results_json}")
        return jsonify(results_json), 200
    
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Batch prediction endpoint
@app.route('/batch_predict', methods=['POST'])
@limiter.limit("5 per minute")
def batch_predict():
    try:
        if 'file' not in request.files:
            logger.error("No file provided in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            logger.error("Invalid file format: must be CSV")
            return jsonify({'error': 'File must be a CSV'}), 400

        df = pd.read_csv(file)
        required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400

        # Convert data types
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Validate numeric inputs
        if df[['tenure', 'MonthlyCharges', 'TotalCharges']].isnull().any().any():
            logger.error("Invalid numeric input after conversion")
            return jsonify({'error': 'Invalid numeric input: tenure, MonthlyCharges, or TotalCharges must be valid numbers'}), 400
        if (df['tenure'] < 0).any() or (df['tenure'] > 120).any():
            logger.error("Tenure validation failed for batch")
            return jsonify({'error': 'Tenure must be between 0 and 120 months'}), 400
        if (df['MonthlyCharges'] <= 0).any() or (df['MonthlyCharges'] > 500).any():
            logger.error("MonthlyCharges validation failed for batch")
            return jsonify({'error': 'MonthlyCharges must be between 0 and 500'}), 400
        if (df['TotalCharges'] <= 0).any() or (df['TotalCharges'] > 10000).any():
            logger.error("TotalCharges validation failed for batch")
            return jsonify({'error': 'TotalCharges must be between 0 and 10000'}), 400

        # Apply feature engineering
        df = calculate_engineered_features(df)

        # Validate engineered features
        if df[['AvgMonthlyCharges', 'CLV', 'ServiceCount', 'ContractRiskFactor']].isnull().any().any():
            logger.error("Computed features contain NaN values for batch")
            return jsonify({'error': 'Computed features (AvgMonthlyCharges, CLV, ServiceCount, or ContractRiskFactor) are invalid'}), 400

        # Make predictions
        results = predict_new_customers(model, df)

        # Save results to CSV
        output = io.StringIO()
        results.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='batch_predictions.csv'
        )
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

# Feature importance endpoint
@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    try:
        # Extract feature names from the preprocessor
        preprocessor = model.named_steps['preprocessor']
        feature_names = []
        # Numeric features
        numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'CLV', 'ServiceCount', 'ContractRiskFactor']
        feature_names.extend(numeric_features)
        # Categorical features (excluding SeniorCitizen since it's numeric)
        categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        
        # Validate the number of categorical features matches the OneHotEncoder's categories
        transformer = preprocessor.named_transformers_['cat']
        ohe = transformer.named_steps['onehot']
        if len(categorical_features) != len(ohe.categories_):
            logger.error(f"Mismatch between expected categorical features ({len(categorical_features)}) and OneHotEncoder categories ({len(ohe.categories_)})")
            return jsonify({'error': 'Mismatch in categorical features during feature importance extraction'}), 500

        for feature in categorical_features:
            try:
                idx = categorical_features.index(feature)
                categories = ohe.categories_[idx]
                feature_names.extend([f"{feature}_{cat}" for cat in categories])
            except IndexError as e:
                logger.error(f"Index error for feature '{feature}' at index {idx}. Categories available: {len(ohe.categories_)}")
                return jsonify({'error': f'Index error while processing feature {feature}: {str(e)}'}), 500

        # Get feature importances from the classifier
        classifier = model.named_steps['classifier']
        importances = classifier.feature_importances_ if hasattr(classifier, 'feature_importances_') else np.abs(classifier.coef_[0])
        
        # Validate lengths match
        if len(feature_names) != len(importances):
            logger.error(f"Length mismatch: feature_names ({len(feature_names)}) vs importances ({len(importances)})")
            return jsonify({'error': 'Mismatch between number of features and importances'}), 500
        
        # Create a DataFrame and sort by importance
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        
        logger.info(f"Feature importance retrieved successfully: {importance_df.to_dict(orient='records')}")
        return jsonify(importance_df.to_dict(orient='records')), 200
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

# Serve the frontend
@app.route('/')
def serve():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
        exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        exit(1)