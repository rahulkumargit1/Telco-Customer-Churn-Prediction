# Telco Churn Analytics Platform

## Overview
The Telco Churn Analytics Platform is a web-based application designed to predict customer churn for a telecommunications company using machine learning. It provides an intuitive interface for both single and batch predictions, along with insights into key factors influencing churn. The platform is built with a Flask backend, a modern frontend using Tailwind CSS and Chart.js, and a robust machine learning pipeline using scikit-learn and imbalanced-learn. This project was created by Rahul Kumar.

## Features
- **Single Prediction**: Input individual customer data through a form to predict churn probability.
- **Batch Prediction**: Upload a CSV file to predict churn for multiple customers and download results.
- **Key Insights**: Visualize critical factors affecting churn, such as contract type, cost, and tenure.
- **Feature Importance**: Display the top features influencing churn predictions.
- **Responsive Design**: Mobile-friendly interface with a collapsible sidebar and dynamic layouts.
- **Rate Limiting**: Ensures fair usage with Flask-Limiter (200 requests/day, 50 requests/hour, 10 single predictions/minute, 5 batch predictions/minute).
- **Error Handling**: Comprehensive logging and user-friendly error messages.
- **Data Validation**: Strict validation of input data to ensure accurate predictions.

## Project Structure
```
telco-churn-analytics/
├── app.py                    # Flask backend for API endpoints
├── churn_prediction.py       # Machine learning pipeline for training and prediction
├── static/
│   └── index.html            # Frontend HTML with Tailwind CSS and JavaScript
├── churn_prediction_results/ # Directory for model and visualization outputs
│   ├── churn_prediction_model.pkl
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── feature_importance.csv
├── web_churn_prediction.log  # Backend logs
├── churn_prediction.log      # ML pipeline logs
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Prerequisites
- Python 3.8+
- Flask
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- joblib
- matplotlib
- seaborn
- flask-limiter
- A CSV dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) for training the model

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rahulkumargit1/telco-churn-analytics.git
   cd telco-churn-analytics
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   - Place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the project root or specify its path via the `INPUT_FILE` environment variable.

5. **Set Environment Variables** (optional):
   ```bash
   export INPUT_FILE="path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv"
   export OUTPUT_DIR="churn_prediction_results"
   ```

## Usage
1. **Train the Model**:
   Run the `churn_prediction.py` script to train the machine learning model:
   ```bash
   python churn_prediction.py
   ```
   This generates the model file (`churn_prediction_model.pkl`) and visualizations in the `churn_prediction_results` directory.

2. **Run the Flask Application**:
   Start the Flask server:
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`.

3. **Access the Platform**:
   - Open a web browser and navigate to `http://localhost:5000`.
   - Use the sidebar to navigate between "Key Insights," "Predict Churn," and "Results Dashboard."
   - For single predictions, fill out the form under "Predict Churn" and submit.
   - For batch predictions, upload a CSV file with the required columns (see below).

4. **Required CSV Columns for Batch Prediction**:
   The CSV file must include:
   - `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

## API Endpoints
- **POST /predict**: Submit a JSON payload with customer data for single prediction.
  - Example:
    ```json
    [{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0
    }]
    ```
  - Response: JSON with `Churn_Status` and `Churn_Probability`.

- **POST /batch_predict**: Upload a CSV file for batch predictions.
  - Response: CSV file (`batch_predictions.csv`) with predictions.

- **GET /feature_importance**: Retrieve the top 10 features influencing churn.
  - Response: JSON array of feature names and importance scores.

- **GET /**: Serve the frontend (`index.html`).

## Model Details
- **Preprocessing**: Handles numeric (scaling, imputation) and categorical (one-hot encoding) features.
- **Feature Engineering**:
  - `AvgMonthlyCharges`: TotalCharges / tenure
  - `CLV`: tenure * MonthlyCharges
  - `ServiceCount`: Number of active services
  - `ContractRiskFactor`: Risk score based on contract type
- **Models Evaluated**: Random Forest, Logistic Regression, Gradient Boosting
- **Best Model**: Selected based on F1 score using GridSearchCV.
- **Class Imbalance**: Handled with SMOTE if imbalance ratio < 0.4.
- **Metrics**: Accuracy, Precision, Recall, F1, ROC AUC, Average Precision.

## Visualizations
Generated during model training and saved in `churn_prediction_results`:
- `churn_distribution.png`: Bar plot of churn distribution.
- `confusion_matrix.png`: Confusion matrix for the best model.
- `roc_curve.png`: ROC curve with AUC score.
- `pr_curve.png`: Precision-Recall curve with AP score.
- `feature_importance.png`: Bar plot of top 15 feature importances.

## Logging
- Backend logs: `web_churn_prediction.log`
- ML pipeline logs: `churn_prediction.log`

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on the GitHub repository: [rahulkumargit1/telco-churn-analytics](https://github.com/rahulkumargit1/telco-churn-analytics).

## License
This project is licensed under the MIT License.

## Contact
For questions or support, contact Rahul Kumar via GitHub: [rahulkumargit1](https://github.com/rahulkumargit1).
