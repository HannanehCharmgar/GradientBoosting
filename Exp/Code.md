### اضافه کردن کتابخانه ها
```
# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
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
```
این سلول همه کتابخانه‌های مورد نیاز پروژه رو وارد می‌کنه.

کتابخانه pandas و numpy برای کار با دیتا فریم و محاسبات عددی استفاده می‌شن.

کتابخانه matplotlib و seaborn برای رسم نمودارها.

بخش warnings.filterwarnings('ignore') برای مخفی کردن هشدارهای غیرضروری.

از sklearn.model_selection برای تقسیم داده و Cross Validation استفاده می‌کنیم.

کتابخانه sklearn.metrics برای محاسبه معیارهای عملکرد مثل Accuracy، ROC-AUC، F1 و Confusion Matrix.

کتابخانه LogisticRegression و StandardScaler برای پیاده‌سازی و آماده‌سازی مدل لجستیک.

کتابخانه CatBoostClassifier برای gradient boosting هست که برای داده‌های ترکیبی (عددی و دسته‌ای) عالیه.

 و sns.set() و plt.rcParams برای تنظیم ظاهر نمودارها.

### بارگذاری و پاکسازی دیتا

```
# Cell 2: Load & Clean Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

print("Dataset shape:", df.shape)
print("Class distribution:\n", df['Churn'].value_counts(normalize=True))
```

دیتاست Telco Customer Churn رو بارگذاری می‌کنیم.

ستون TotalCharges بعضاً رشته است، پس به عدد تبدیل می‌کنیم و هر ردیفی که مشکل داشته باشه حذف می‌کنیم.

ستون هدف Churn به عدد تبدیل میشه: Yes=1, No=0.

ستون customerID حذف میشه چون برای مدل اهمیتی نداره.

در نهایت ابعاد دیتاست و توزیع کلاس‌ها چاپ میشه تا متوجه imbalance شویم.

### تقسیم داده TRAIN/TEST

```
# Cell 3: Train/Test Split
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical features
cat_features = X.select_dtypes(include=['object']).columns.tolist()
cat_indices = [X.columns.get_loc(col) for col in cat_features]

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```
داده‌ها به ویژگی‌ها (X) و هدف (y) تقسیم میشن.

ویژگی‌های دسته‌ای شناسایی میشن تا CatBoost بدونه کدوم ستون‌ها categorical هستند.

با train_test_split داده‌ها به Train 80% و Test 20% تقسیم میشن و stratify=y تضمین می‌کنه نسبت churn و non-churn در Train و Test حفظ بشه.

### مدل CatBoost 

```
# Cell 4: CatBoost Model Training
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
```
بخش class_weights برای رسیدگی به imbalance استفاده میشه.

مدل CatBoost با iterations=800, depth=6, learning_rate=0.05 آموزش داده میشه.

بخش eval_set و early_stopping_rounds باعث میشن اگر مدل بعد از 50 iteration بهتر نشه، آموزش متوقف بشه.

بخش use_best_model=True تضمین می‌کنه بهترین مدل روی Test ذخیره بشه.

### ارزیابی مدل CatBoost
```
# Cell 5: CatBoost Predictions & Metrics
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
```
مدل CatBoost روی داده Test پیش‌بینی می‌کنه.

بخش y_proba_cat احتمال Churn رو میده.

معیارهای مهم: Accuracy, ROC-AUC, PR-AUC, F1 محاسبه میشن.

این معیارها کیفیت پیش‌بینی مدل رو نشون میدن و برای مقایسه با Logistic Regression آماده هستن.

### اعتبار سنجی مدل

```
# Cell 6: Manual Cross Validation
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
```
اعتبارسنجی یا Cross Validation دستی برای جلوگیری از ارور sklearn clone انجام شد.

داده‌ها به ۵ fold تقسیم میشن و مدل روی هر fold آموزش و ارزیابی میشه.

پارامتر roc_auc_score معیار اصلی هست.

نتیجه: Mean AUC میانگین عملکرد و Std AUC پایداری مدل رو نشون میده.

### یافتن Threshold بهینه

```
# Cell 7: Threshold Optimization
thresholds = np.linspace(0.1, 0.9, 100)
f1_scores = []

for t in thresholds:
    preds = (y_proba_cat >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds))

best_threshold = thresholds[np.argmax(f1_scores)]
print("\nBest Threshold (F1 Optimized):", round(best_threshold,3))
```
ه طور پیش‌فرض threshold=0.5 برای Churn هست.

این سلول threshold بهینه برای بیشینه کردن F1 Score پیدا می‌کنه.

نتیجه: اگر احتمال > 0.625 باشه → پیش‌بینی Churn.

### ماتریس درهم ریختگی 

```
# Cell 8: Confusion Matrix 
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
```
این ماتریس نمایش میده که مدل روی هر کلاس چقدر درست پیش‌بینی کرده.

سلول‌ها هم عدد مطلق و هم درصد نمایش داده میشن.

دسته های No Churn و Churn به صورت واضح روی محورهای x و y مشخص شدن.

### بررسی اهمیت Feature ها 

```
# Cell 9: Feature Importance
importances = cat_model.get_feature_importance()
fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(8,6))
sns.barplot(data=fi_df, x="Importance", y="Feature")
plt.title("Top 15 Feature Importances - CatBoost")
plt.show()
```
این سلول ویژگی‌های مهم مدل CatBoost رو مشخص می‌کنه.

بخش get_feature_importance() میزان تاثیر هر ویژگی رو در پیش‌بینی میده.

فقط ۱۵ ویژگی برتر نمایش داده میشه.

### مدل رگرسیون لجستیک 

```
# Cell 10: Logistic Regression Training
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
```
داده‌ها categorical → one-hot encode میشن.

ویژگی‌ها استانداردسازی میشن (Logistic Regression به مقیاس حساسه).

مدل آموزش داده میشه و پیش‌بینی و معیارهای عملکرد محاسبه میشن.

### نمودار ROC 

```
# Cell 11: ROC Curve Comparison
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
```
نمودار ROC Curve تفکیک بین کلاس‌ها رو نشون میده.

 مدل های CatBoost و Logistic با هم مقایسه میشن.

خط خطی (diagonal) = حدس تصادفی. هرچی منحنی بالاتر باشه، عملکرد بهتره.

### نمودار Precision-Recall 
```
# Cell 12: Precision-Recall Curve Comparison
plt.figure(figsize=(7,6))

prec_cat, rec_cat, _ = precision_recall_curve(y_test, y_proba_cat)
prec_log, rec_log, _ = precision_recall_curve(y_test, y_proba_log)

plt.plot(rec_cat, prec_cat, label=f"CatBoost (PR-AUC={pr_auc_cat:.3f})")
plt.plot(rec_log, rec_log, label=f"Logistic (PR-AUC={pr_auc_log:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.show()
```
نمودار Precision-Recall Curve برای دیتاست‌های imbalance مهمه.

هرچقدر PR-AUC عدد بزرگتری باشد = بهتر بودن مدل برای تشخیص Churn.

مدل CatBoost معمولاً PR-AUC بالاتری داره.

### جدول عملکرد 2 مدل ( CatBoost & LogisticRegression )

```
# Cell 13: Final Model Comparison Table
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "CatBoost"],
    "Accuracy": [acc_log, acc_cat],
    "ROC-AUC": [auc_log, auc_cat],
    "PR-AUC": [pr_auc_log, pr_auc_cat],
    "F1-Score": [f1_log, f1_cat]
})

print("\n===== Final Model Comparison =====")
print(comparison)
```
جدول نهایی عملکرد مدل‌ها.

شامل Accuracy، ROC-AUC، PR-AUC و F1-Score برای CatBoost و Logistic.

