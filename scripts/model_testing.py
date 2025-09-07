from utils import load_csv, split_xy, check_missing_and_duplicates
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def test_model(model_path='./models/xgb_model.pkl', 
               test_path='./data/split/test.csv',
               threshold=0.90):
    """Simple model testing function"""
    
    print("🚀 Loading model and data...")
    model = joblib.load(model_path)
    df = load_csv(test_path)
    
    # Clean data
    _, df = check_missing_and_duplicates(df, drop_duplicates=True)
    X_test, y_test = split_xy(df, 'Class')
    
    print(f"📊 Test data: {df.shape[0]} samples")
    print(f"📈 Fraud cases: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
    
    print("\n🔍 Making predictions...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    print(f"✅ Using threshold: {threshold}")
    print(f"📊 Predicted fraud: {sum(y_pred)} cases")
    
    # Key metrics
    print("\n📈 Results:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    print(f"   ROC-AUC: {auc:.3f}")
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Pos: {tp}, False Pos: {fp}")
    print(f"   False Neg: {fn}, True Neg: {tn}")
    
    # Feature importance
    print("\n🔝 Top 10 Features:")
    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps['model']
    else:
        actual_model = model
        

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Return results for further analysis
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        }
    }

def quick_threshold_test(results, thresholds=[0.5, 0.7, 0.8, 0.87, 0.9]):
    """Quick threshold comparison"""
    print("\n🔧 Threshold Comparison:")
    print("Threshold | Precision | Recall | F1-Score")
    print("-" * 40)
    
    y_test = results['y_test']
    y_proba = results['y_proba']
    
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{t:8.2f} | {precision:8.3f} | {recall:6.3f} | {f1:8.3f}")

if __name__ == "__main__":

    results = test_model()
    

    quick_threshold_test(results)
    
    print("\n✅ Testing complete!")