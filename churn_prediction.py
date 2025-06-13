import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, 
                             classification_report, confusion_matrix, roc_curve, precision_recall_curve, 
                             average_precision_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
from os import environ

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("churn_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("churn_prediction")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define valid categorical values
VALID_CATEGORIES = {
    'gender': ['Male', 'Female'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['Fiber optic', 'DSL', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

def load_and_clean_data(file_path):
    """
    Load and preprocess the Telco Customer Churn dataset.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset from {file_path} with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_idx = df['TotalCharges'].isnull()
    df.loc[missing_idx, 'TotalCharges'] = df.loc[missing_idx, 'MonthlyCharges']
    
    # Convert Churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def explore_data(df):
    """
    Explore and visualize the dataset.
    """
    logger.info(f"Dataset Preview:\n{df.head()}")
    logger.info(f"\nDataset Summary Statistics:\n{df.describe()}")
    churn_dist = df['Churn'].value_counts(normalize=True)
    logger.info(f"\nChurn Distribution:\n{churn_dist}")
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=churn_dist.index, y=churn_dist.values)
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Proportion')
    plt.savefig('churn_distribution.png', dpi=300)
    plt.close()
    
    return churn_dist[1] / churn_dist[0]

def create_features(df):
    """
    Create new features for prediction.
    """
    df_new = df.copy()
    
    # Validate categorical features
    for col, valid_values in VALID_CATEGORIES.items():
        if col in df_new.columns:
            invalid = df_new[col][~df_new[col].isin(valid_values)]
            if not invalid.empty:
                raise ValueError(f"Invalid values in {col}: {invalid.unique()}")
    
    # Calculate average monthly charges
    df_new['AvgMonthlyCharges'] = df_new['TotalCharges'] / np.maximum(df_new['tenure'], 1)
    
    # Customer Lifetime Value (CLV)
    df_new['CLV'] = df_new['tenure'] * df_new['MonthlyCharges']
    
    # Service Count
    service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_columns:
        df_new[col + '_Numeric'] = df_new[col].apply(lambda x: 1 if x == 'Yes' else 0)
    df_new['ServiceCount'] = df_new[[col + '_Numeric' for col in service_columns]].sum(axis=1)
    df_new.drop([col + '_Numeric' for col in service_columns], axis=1, inplace=True)
    
    # Contract risk factor
    contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
    df_new['ContractRiskFactor'] = df_new['Contract'].map(contract_risk)
    
    logger.info("Created features: AvgMonthlyCharges, CLV, ServiceCount, ContractRiskFactor")
    return df_new

def prepare_data(df):
    """
    Prepare data for modeling.
    """
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    y = df['Churn']
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Feature split - Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}")
    return X, y, numeric_features, categorical_features

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Create preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def get_feature_names(pipeline, numeric_features, categorical_features):
    """
    Extract feature names from the pipeline.
    """
    try:
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
        return numeric_features + cat_feature_names
    except Exception as e:
        logger.error(f"Error getting feature names: {str(e)}")
        return numeric_features + categorical_features

def get_feature_importance(pipeline, feature_names):
    """
    Extract feature importance.
    """
    try:
        classifier = pipeline.named_steps.get('classifier', None)
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            return None
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return None

def generate_visualizations(y_test, y_pred, y_pred_proba, model_name, output_dir):
    """
    Generate evaluation visualizations.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300)
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision_score(y_test, y_pred_proba):.4f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', label='Random')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=300)
    plt.close()

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, class_imbalance, numeric_features, categorical_features, output_dir):
    """
    Train and evaluate models.
    """
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 10],
                'classifier__class_weight': [None, 'balanced']
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l2'],
                'classifier__class_weight': [None, 'balanced']
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        }
    }
    
    use_smote = class_imbalance < 0.4
    results = {}
    best_score = 0
    best_model_name = None
    best_model_pipeline = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model_info in models.items():
        logger.info(f"\nTraining {model_name}...")
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)) if use_smote else ('nosmote', 'passthrough'),
            ('classifier', model_info['model'])
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            model_info['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
        
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'cv_f1': cross_validate(best_estimator, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), 
                                    cv=cv, scoring='f1')['test_score'].mean(),
            'best_params': grid_search.best_params_,
            'pipeline': best_estimator,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"{model_name} Results:\nBest Parameters: {results[model_name]['best_params']}")
        logger.info(f"F1 Score: {results[model_name]['f1']:.4f}")
        
        if results[model_name]['f1'] > best_score:
            best_score = results[model_name]['f1']
            best_model_name = model_name
            best_model_pipeline = best_estimator
    
    logger.info(f"\nBest Model: {best_model_name} with F1 Score: {best_score:.4f}")
    generate_visualizations(y_test, results[best_model_name]['y_pred'], 
                           results[best_model_name]['y_pred_proba'], best_model_name, output_dir)
    
    feature_names = get_feature_names(best_model_pipeline, numeric_features, categorical_features)
    feature_importance = get_feature_importance(best_model_pipeline, feature_names)
    if feature_importance is not None:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Feature Importances - {best_model_name}')
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
        plt.close()
        feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        logger.info(f"\nTop 15 Feature Importances:\n{feature_importance.head(15)}")
    
    return best_model_pipeline

def predict_new_customers(model, new_data):
    """
    Predict churn for new customers.
    """
    try:
        # Validate required columns
        required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'CLV',
            'ServiceCount', 'ContractRiskFactor'
        ]
        missing_cols = [col for col in required_columns if col not in new_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Validate data types
        new_data['SeniorCitizen'] = pd.to_numeric(new_data['SeniorCitizen'], errors='coerce')
        new_data['tenure'] = pd.to_numeric(new_data['tenure'], errors='coerce')
        new_data['MonthlyCharges'] = pd.to_numeric(new_data['MonthlyCharges'], errors='coerce')
        new_data['TotalCharges'] = pd.to_numeric(new_data['TotalCharges'], errors='coerce')
        new_data['AvgMonthlyCharges'] = pd.to_numeric(new_data['AvgMonthlyCharges'], errors='coerce')
        new_data['CLV'] = pd.to_numeric(new_data['CLV'], errors='coerce')
        new_data['ServiceCount'] = pd.to_numeric(new_data['ServiceCount'], errors='coerce')
        new_data['ContractRiskFactor'] = pd.to_numeric(new_data['ContractRiskFactor'], errors='coerce')
        
        # Check for NaN values after conversion
        invalid_cols = new_data[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                                 'AvgMonthlyCharges', 'CLV', 'ServiceCount', 'ContractRiskFactor']].isnull().any()
        if invalid_cols.any():
            invalid_cols = invalid_cols[invalid_cols].index.tolist()
            raise ValueError(f"Invalid numeric values in columns: {invalid_cols}")
        
        # Validate numerical ranges
        if (new_data['tenure'] < 0).any():
            raise ValueError("Tenure cannot be negative")
        if (new_data['MonthlyCharges'] <= 0).any():
            raise ValueError("MonthlyCharges must be positive")
        if (new_data['TotalCharges'] <= 0).any():
            raise ValueError("TotalCharges must be positive")
        
        # Validate categorical values
        for col, valid_values in VALID_CATEGORIES.items():
            if col in new_data.columns:
                invalid = new_data[col][~new_data[col].isin(valid_values)]
                if not invalid.empty:
                    raise ValueError(f"Invalid values in {col}: {invalid.unique()}")
        
        # Make predictions
        prediction = model.predict(new_data)
        probability = model.predict_proba(new_data)[:, 1]
        
        result_df = new_data.copy()
        result_df['Churn_Prediction'] = prediction
        result_df['Churn_Probability'] = probability
        result_df['Churn_Status'] = result_df['Churn_Prediction'].apply(lambda x: 'Yes' if x == 1 else 'No')
        
        for i, row in result_df.iterrows():
            logger.info(f"Customer {i+1} Prediction: Churn Status: {row['Churn_Status']}, Probability: {row['Churn_Probability']:.4f}")
        
        return result_df[['Churn_Status', 'Churn_Probability']]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\nInput data: {new_data.to_dict()}")
        raise

def main(file_path, output_dir):
    """
    Run the churn prediction pipeline.
    """
    logger.info("Starting churn prediction pipeline...")
    
    df = load_and_clean_data(file_path)
    class_imbalance = explore_data(df)
    df = create_features(df)
    X, y, numeric_features, categorical_features = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, class_imbalance, 
                                    numeric_features, categorical_features, output_dir)
    
    model_path = f"{output_dir}/churn_prediction_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved as '{model_path}'")
    
    return best_model

if __name__ == "__main__":
    input_file = environ.get('INPUT_FILE', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    output_dir = environ.get('OUTPUT_DIR', 'churn_prediction_results')
    main(input_file, output_dir)