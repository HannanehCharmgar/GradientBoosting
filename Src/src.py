# ========================
# 1. Import Libraries
# ========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]


# ========================
# 2. Load & Clean Dataset
# ========================
df = pd.read_csv("Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop('customerID', axis=1, inplace=True)

print("Dataset shape:", df.shape)
print("Class distribution:\n", df['Churn'].value_counts(normalize=True))


# ========================
# 3. Train/Test Split
# ========================
X = df.drop('Churn', axis=1)
y = df['Churn']

cat_features = X.select_dtypes(include=['object']).columns.tolist()
cat_indices = [X.columns.get_loc(col) for col in cat_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# ==================== CATBOOST MODEL ======================
# ==========================================================

# --- Calculate class weight safely (convert to pure float)
pos_weight = float(y_train.value_counts()[0] / y_train.value_counts()[1])

cat_model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_state=42,
    class_weights=[1.0, pos_weight],
    verbose=0
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_indices,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    use_best_model=True
)

# Predictions
y_pred_cat = cat_model.predict(X_test)
y_proba_cat = cat_model.predict_proba(X_test)[:, 1]

# Metrics
acc_cat = accuracy_score(y_test, y_pred_cat)
auc_cat = roc_auc_score(y_test, y_proba_cat)
f1_cat = f1_score(y_test, y_pred_cat)
pr_auc_cat = average_precision_score(y_test, y_proba_cat)

print("\n===== CatBoost Performance =====")
print("Accuracy :", round(acc_cat, 4))
print("ROC-AUC  :", round(auc_cat, 4))
print("PR-AUC   :", round(pr_auc_cat, 4))
print("F1-Score :", round(f1_cat, 4))


# ========================
# Cross Validation 
# ========================

print("\nRunning Manual Cross Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    model_cv = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        verbose=0
    )
    
    model_cv.fit(
        X_train_cv,
        y_train_cv,
        cat_features=cat_indices
    )
    
    y_val_proba = model_cv.predict_proba(X_val_cv)[:, 1]
    fold_auc = roc_auc_score(y_val_cv, y_val_proba)
    
    cv_auc_scores.append(fold_auc)
    
    print(f"Fold {fold+1} AUC: {round(fold_auc,4)}")

print("\nMean CV AUC:", round(np.mean(cv_auc_scores),4))
print("Std CV AUC :", round(np.std(cv_auc_scores),4))
# ========================
# Threshold Optimization
# ========================
thresholds = np.linspace(0.1, 0.9, 100)
f1_scores = []

for t in thresholds:
    preds = (y_proba_cat >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds))

best_threshold = thresholds[np.argmax(f1_scores)]
print("\nBest Threshold (F1 Optimized):", round(best_threshold,3))


# ========================
# Confusion Matrix 
# ========================
cm = confusion_matrix(y_test, y_pred_cat)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels = np.array([
    [f"{cm[i][j]}\n({cm_percent[i][j]:.1%})"
     for j in range(cm.shape[1])]
    for i in range(cm.shape[0])
])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
            xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"])
plt.title("Confusion Matrix - CatBoost")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# ========================
# Feature Importance
# ========================
importances = cat_model.get_feature_importance()

fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(8,6))
sns.barplot(data=fi_df, x="Importance", y="Feature")
plt.title("Top 15 Feature Importances - CatBoost")
plt.show()


# ==========================================================
# ================ LOGISTIC REGRESSION =====================
# ==========================================================

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

log_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_proba_log = log_model.predict_proba(X_test_scaled)[:,1]

acc_log = accuracy_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_proba_log)
f1_log = f1_score(y_test, y_pred_log)
pr_auc_log = average_precision_score(y_test, y_proba_log)


# ========================
# ROC Comparison
# ========================
plt.figure(figsize=(7,6))

fpr_cat, tpr_cat, _ = roc_curve(y_test, y_proba_cat)
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)

plt.plot(fpr_cat, tpr_cat, label=f"CatBoost (AUC={auc_cat:.3f})")
plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC={auc_log:.3f})")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
#==================================================
# Precision-Recall Curve Comparison (Corrected)
from sklearn.metrics import precision_recall_curve, average_precision_score

# CatBoost
prec_cat, rec_cat, _ = precision_recall_curve(y_test, y_proba_cat)
pr_auc_cat = average_precision_score(y_test, y_proba_cat)

# Logistic Regression
prec_log, rec_log, _ = precision_recall_curve(y_test, y_proba_log)
pr_auc_log = average_precision_score(y_test, y_proba_log)

# Plot
plt.figure(figsize=(8,6))
plt.plot(rec_cat, prec_cat, label=f"CatBoost (PR-AUC={pr_auc_cat:.3f})", color="#1f77b4", linewidth=2)
plt.plot(rec_log, prec_log, label=f"Logistic (PR-AUC={pr_auc_log:.3f})", color="#ff7f0e", linewidth=2)
plt.fill_between(rec_cat, prec_cat, alpha=0.1, color="#1f77b4")
plt.fill_between(rec_log, prec_log, alpha=0.1, color="#ff7f0e")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# Cell 13: Final Model Comparison Table
# ====================================================
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "CatBoost"],
    "Accuracy": [acc_log, acc_cat],
    "ROC-AUC": [auc_log, auc_cat],
    "PR-AUC": [pr_auc_log, pr_auc_cat],
    "F1-Score": [f1_log, f1_cat]
})

print("\n===== Final Model Comparison =====")
print(comparison)
