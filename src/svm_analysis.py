import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA # Για ταχύτητα
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def prepare_data_from_csv(csv_path):
    print(f"\nLOADING DATA from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: File not found at {csv_path}")
        sys.exit(1)  # raise error kalutera me description 

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Μετατροπή σε string για να αποφύγουμε κρασαρίσματα
    df['sample_code'] = df['sample_code'].astype(str)

    metadata_cols = ['id', 'sample_code', 'botanical', 'geographic']
    X = df.drop(columns=metadata_cols, errors='ignore').values
    
    #  5 βασικές κατηγορίες
    valid_classes = ['thymari', 'pefko', 'eriki', 'ilianthos', 'vamvaki']
    valid_indices = df['botanical'].isin(valid_classes)

    df = df[valid_indices]
    X = X[valid_indices]
    y = df['botanical'].values 
    groups = df['sample_code'].values
    
    print("-" * 30)
    print(f"Data Loaded Successfully!")
    print(f"Total Samples: {X.shape[0]}")
    print(f"Features (Wavelengths): {X.shape[1]}")
    print(f"Unique Groups (Samples): {len(np.unique(groups))}")
    print("-" * 30)
    
    return X, y, groups

def run_svm_analysis(X, y, groups):
    print("\nΕΚΚΙΝΗΣΗ 5-FOLD STRATIFIED GROUP CROSS-VALIDATION...")
    
   
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    #  μοντελα
    svm_models = {
        'Linear Kernel': SVC(kernel='linear', C=1, class_weight='balanced'),
        'RBF Kernel': SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced'),
        'Poly Kernel (Deg 3)': SVC(kernel='poly', degree=3, C=1, class_weight='balanced')# γιατι 3ου βαθμου ?
    }

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Κύριος Βρόχος: Για κάθε Μοντέλο
    for name, model in svm_models.items():
        print(f"\n{'='*20} {name} {'='*20}")
        
        fold_accuracies = []
        
        # Λίστεσ για να μαζέψουμε ολεσ τις προβλεψεισ και τισ αληθινεσ τιμεσ 
        all_y_true = []
        all_y_pred = []

        # Για κάθε Fold
        for i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=50)), 
                ('svm', model)
            ])
            
            # εκπαιδευση
            pipeline.fit(X_train, y_train)
            
            # προβλεψη
            y_pred = pipeline.predict(X_test)
            
            # Αποθήκευση αποτελεσμάτων αυτού του fold
            acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(acc)
            
            # labels για το ρεπορτ 
            all_y_true.extend(y_test)  # extend γιατι θελουμε η πληροφορια να μπει σε μια λιστα , δεν θελουμε λιστα με λιστες
            all_y_pred.extend(y_pred)
            
            print(f"Fold {i+1}/5 -> Accuracy: {acc:.4f}")

        # --- ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ ---
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        
        print(f"\n>>> ΤΕΛΙΚΗ ΑΚΡΙΒΕΙΑ (Mean): {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Classification Report (Συνολικό για όλα τα folds)
        report_str = classification_report(all_y_true, all_y_pred, zero_division=0)
        print("\n--- Aggregate Classification Report ---")
        print(report_str)
        
        # Αποθήκευση σε TXT
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        txt_filename = f"report_CV_{safe_name}.txt"
        
        with open(txt_filename, "w", encoding='utf-8') as f:
            f.write(f"MODEL: {name} (5-Fold CV)\n")
            f.write(f"MEAN ACCURACY: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
            f.write("-" * 30 + "\n")
            f.write("Fold Scores: " + str([round(x,4) for x in fold_accuracies]) + "\n")
            f.write("=" * 60 + "\n")
            f.write(report_str)
            
        print(f" Report saved: {txt_filename}")

        # ---  CONFUSION MATRIX ---
        cm = confusion_matrix(all_y_true, all_y_pred)
        unique_labels = np.unique(y)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f'Aggregate Confusion Matrix - {name}\n(Mean Acc: {mean_acc:.2%})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = f'plots/confusion_matrix_CV_{safe_name}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f" Plot saved: {plot_path}")

if __name__ == "__main__":
    file_path = r"C:\Users\egiannikos\Desktop\dataset_classification_pipeline\dataset_classification_pipeline-main\data\processed\spectral_dataset_enriched.csv"
    
    # Εκτέλεση
    X, y, groups = prepare_data_from_csv(file_path)
    run_svm_analysis(X, y, groups)