# 🚨 Smarter Fraud Detection

A comprehensive machine learning project for detecting fraudulent transactions using advanced feature engineering and XGBoost classification.

**Author:** Ahmed Abdel Baqi  
**LinkedIn:** [ahmed-abdel-baqi-931b29338](https://www.linkedin.com/in/ahmed-abdel-baqi-931b29338/)  
**GitHub:** [Ahmed249323](https://github.com/Ahmed249323)

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a sophisticated fraud detection system that leverages machine learning techniques to identify fraudulent transactions. The system uses XGBoost with advanced preprocessing, SMOTE for handling class imbalance, and threshold optimization for optimal performance.

### 🎯 Project Goals
- Build a robust fraud detection model with high precision and recall
- Handle class imbalance in financial transaction data
- Implement end-to-end ML pipeline from data cleaning to model deployment
- Provide comprehensive evaluation metrics and visualizations

### ✨ Key Highlights

- **Advanced Feature Engineering**: Custom features and transformations for better model performance
- **Class Imbalance Handling**: SMOTE oversampling technique to address fraud rarity
- **Robust Preprocessing**: RobustScaler + MinMaxScaler pipeline for optimal scaling
- **Threshold Optimization**: Custom threshold tuning (0.872553) for optimal precision-recall balance
- **Comprehensive Evaluation**: Multiple metrics including ROC-AUC, F1-score, and confusion matrix analysis
- **Production-Ready Code**: Modular scripts with proper error handling and documentation

## 📁 Project Structure

```
Smarter Fraud Detection/
├── models/
│   └── xgb_model.pkl          # Trained XGBoost model
├── notebooks/
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_exploratory_data_analysis(EDA).ipynb
│   └── 04_model_selection _evaluation.ipynb
├── scripts/
│   ├── model_training.py      # Model training pipeline
│   ├── model_testing.py       # Model evaluation and testing
│   └── utils.py              # Utility functions
├── project_packages.txt      # Python dependencies
└── README.md                # This file
```

## ✨ Features

### Data Processing
- **Missing Value Handling**: Comprehensive missing value detection and treatment
- **Duplicate Removal**: Automatic duplicate detection and removal
- **Data Validation**: Quality checks and data integrity validation

### Feature Engineering
- **Advanced Transformations**: Custom feature creation and engineering
- **Scaling**: RobustScaler followed by MinMaxScaler for optimal performance
- **Feature Selection**: Intelligent feature selection and importance analysis

### Model Training
- **XGBoost Classifier**: State-of-the-art gradient boosting
- **SMOTE Oversampling**: Synthetic Minority Oversampling Technique
- **Pipeline Architecture**: Modular and reproducible training pipeline
- **Threshold Optimization**: Custom threshold tuning (0.872553)

### Model Evaluation
- **Comprehensive Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrix**: Visual representation of model performance
- **Threshold Analysis**: Multiple threshold comparison
- **Feature Importance**: Top feature analysis and visualization

## 🚀 Installation

### Prerequisites

- Python 3.13.5
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Smarter Fraud Detection"
   ```

2. **Install dependencies**
   ```bash
   pip install -r project_packages.txt
   ```

### Dependencies

```
pandas==2.3.2
numpy==2.2.0
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.7.1
joblib==1.4.2
imbalanced-learn==0.14.0
xgboost==3.0.2
```

## 💻 Usage

### Training the Model

```python
# Run the training script
python scripts/model_training.py
```

The training script will:
1. Load and clean the training data
2. Apply SMOTE for class balancing
3. Train the XGBoost model with optimized parameters
4. Save the trained model to `models/xgb_model.pkl`

### Testing the Model

```python
# Run the testing script
python scripts/model_testing.py
```

The testing script provides:
- Model performance evaluation
- Confusion matrix visualization
- Threshold comparison analysis
- Feature importance ranking

### Using the Utils Module

```python
from scripts.utils import (
    load_csv, 
    split_xy, 
    check_missing_and_duplicates,
    apply_smote,
    train_on_train_with_threshold
)

# Load data
df = load_csv('path/to/data.csv')

# Check data quality
report, clean_df = check_missing_and_duplicates(df, drop_duplicates=True)

# Split features and target
X, y = split_xy(clean_df, target_col='Class')

# Apply SMOTE
X_resampled, y_resampled = apply_smote(X, y)
```

## 🔬 Methodology

### 1. Data Understanding and Cleaning
- Exploratory data analysis
- Missing value analysis
- Duplicate detection and removal
- Data quality assessment

### 2. Feature Engineering
- Custom feature creation
- Statistical transformations
- Feature scaling and normalization
- Feature selection and importance analysis

### 3. Exploratory Data Analysis (EDA)
- Distribution analysis
- Correlation analysis
- Outlier detection
- Class imbalance visualization

### 4. Model Selection and Evaluation
- Multiple algorithm comparison
- Hyperparameter tuning
- Cross-validation
- Performance metric evaluation

### Model Architecture

```python
Pipeline([
    ("robust", RobustScaler()),
    ("minmax", MinMaxScaler()),
    ("model", XGBClassifier(
        scale_pos_weight=10,
        random_state=42,
        eval_metric='logloss',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.2
    ))
])
```

## 📊 Results

### 🎯 Model Performance
- **Optimized Threshold**: 0.872553 (tuned for optimal precision-recall balance)
- **Class Balancing**: SMOTE oversampling technique
- **Scaling Pipeline**: RobustScaler + MinMaxScaler for robust feature scaling
- **Model Architecture**: XGBoost with optimized hyperparameters

### 📈 Key Metrics
The model provides comprehensive evaluation including:
- **Precision**: Measures accuracy of positive predictions
- **Recall**: Measures ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC Score**: Area under the receiver operating characteristic curve
- **Confusion Matrix Analysis**: Detailed breakdown of predictions
- **Feature Importance Ranking**: Understanding which features drive predictions

### 📊 Visualization Features
- **Confusion Matrix Heatmaps**: Visual representation of model performance
- **ROC and Precision-Recall Curves**: Performance across different thresholds
- **Feature Importance Plots**: Understanding model decision-making
- **Threshold Comparison Charts**: Performance analysis across different thresholds

### 🚀 Business Impact
- **Fraud Detection**: Identifies fraudulent transactions with high accuracy
- **Cost Reduction**: Minimizes false positives to reduce operational costs
- **Risk Management**: Provides confidence scores for transaction monitoring
- **Scalability**: Production-ready pipeline for real-time fraud detection

## 🛠️ Development

### Running Notebooks
1. Start Jupyter Notebook
2. Navigate to the `notebooks/` directory
3. Run notebooks in sequence:
   - `01_data_understanding_and_cleaning.ipynb`
   - `02_feature_engineering.ipynb`
   - `03_exploratory_data_analysis(EDA).ipynb`
   - `04_model_selection _evaluation.ipynb`

### Customization
- Modify model parameters in `scripts/model_training.py`
- Adjust threshold values in `scripts/model_testing.py`
- Add new utility functions in `scripts/utils.py`

## 🤝 Contributing

We welcome contributions to improve this fraud detection system! Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### 🎯 Areas for Contribution
- **Feature Engineering**: New feature creation and selection techniques
- **Model Improvements**: Alternative algorithms and ensemble methods
- **Performance Optimization**: Code optimization and parallel processing
- **Documentation**: Improved documentation and tutorials
- **Visualization**: Enhanced plotting and dashboard creation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Connect

**Ahmed Abdel Baqi**  
*Data Scientist & Machine Learning Engineer*

- **LinkedIn**: [ahmed-abdel-baqi-931b29338](https://www.linkedin.com/in/ahmed-abdel-baqi-931b29338/)
- **GitHub**: [Ahmed249323](https://github.com/Ahmed249323)
- **Kaggle**: [ahmedabdelbaqi](https://www.kaggle.com/ahmedabdelbaqi)

For questions, suggestions, or collaboration opportunities, please:
- Open an issue in the repository
- Connect with me on LinkedIn
- Check out my other projects on GitHub

---

## 🌟 Acknowledgments

- Special thanks to the open-source community for the amazing libraries used in this project
- Credit to the financial fraud detection research community for best practices
- Inspired by real-world fraud detection challenges in the financial industry

**Note**: This project is designed for educational and research purposes. Always ensure proper validation and testing before deploying in production environments. The model should be regularly retrained with new data to maintain optimal performance.
