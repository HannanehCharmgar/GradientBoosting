```
# Cell 2: Import Libraries
# ====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# sklearn libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

# CatBoost library
from catboost import CatBoostClassifier

# Initial settings
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

print("✅ All libraries imported successfully!")
```
```
# Cell 3: Load Dataset
# ====================================================
# Load data
df = pd.read_csv("Telco-Customer-Churn.csv")

print("📊 Dataset Overview:")
print("=" * 50)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Column names: {list(df.columns)}")

# Show first 5 rows
print("\n🔍 First 5 rows of dataset:")
print("=" * 50)
print(df.head())

# Statistical information
print("\n📈 Statistical summary of numerical columns:")
print("=" * 50)
print(df.describe())

# Check for null values
print("\n⚠️ Checking for missing values:")
print("=" * 50)
null_info = df.isnull().sum()
print(null_info[null_info > 0] if null_info.sum() > 0 else "✅ No null values found")

# Check data types
print("\n🔧 Data types:")
print("=" * 50)
print(df.dtypes.value_counts())
```

```
# Cell 4: Data Cleaning & Preprocessing
# ====================================================
print("🧹 Starting data cleaning process...")
print("=" * 50)

# 1. Convert TotalCharges to numeric (with error handling)
original_shape = df.shape
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"1. TotalCharges column converted to numeric.")

# 2. Check and remove null values
null_count = df['TotalCharges'].isnull().sum()
if null_count > 0:
    print(f"   ⚠️ Found {null_count} null values in TotalCharges.")
    df.dropna(inplace=True)
    print(f"   ✅ Null values removed.")
else:
    print(f"   ✅ No null values found.")

print(f"   Data volume changes: {original_shape[0]} → {df.shape[0]} rows")

# 3. Convert target variable to numeric (0 and 1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("2. Churn variable converted to numeric: Yes=1, No=0")

# 4. Remove customer ID (not suitable for modeling)
df.drop('customerID', axis=1, inplace=True)
print("3. customerID column removed (IDs are not used in modeling)")

# 5. Check class distribution
print("\n📊 Final class distribution:")
print("=" * 50)
churn_counts = df['Churn'].value_counts()
churn_percent = df['Churn'].value_counts(normalize=True) * 100

for value, count, percent in zip(churn_counts.index, churn_counts.values, churn_percent.values):
    label = "Churn" if value == 1 else "No Churn"
    print(f"{label}: {count} samples ({percent:.1f}%)")

print(f"\n📈 Class ratio: 1:{churn_counts[0]/churn_counts[1]:.1f}")
print("⚠️ Note: Dataset is imbalanced (minority class less than 30%)")
```
```
# Cell 5: Exploratory Data Analysis - Part 1
# ====================================================
print("🔬 Exploratory Data Analysis (EDA)")
print("=" * 50)

# Create large figure for all plots
fig = plt.figure(figsize=(20, 15))

# 1. Target variable distribution
ax1 = plt.subplot(3, 3, 1)
colors = ['#3498db', '#e74c3c']
churn_labels = ['No Churn', 'Churn']
churn_counts = df['Churn'].value_counts()

bars = ax1.bar(churn_labels, churn_counts, color=colors, alpha=0.8)
ax1.set_title('Target Variable Distribution (Churn)', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('Number of Customers')
ax1.grid(axis='y', alpha=0.3)

# Add numbers on bars
for bar, count in zip(bars, churn_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 10, 
             f'{count}\n({count/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=11)

# 2. Tenure distribution
ax2 = plt.subplot(3, 3, 2)
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, ax=ax2, palette=colors)
ax2.set_title('Tenure Distribution by Churn', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tenure (months)')
ax2.set_ylabel('Count')

# 3. MonthlyCharges distribution
ax3 = plt.subplot(3, 3, 3)
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, ax=ax3, palette=colors)
ax3.set_title('Monthly Charges Distribution by Churn', fontsize=14, fontweight='bold')
ax3.set_xlabel('Monthly Charges ($)')
ax3.set_ylabel('Count')

# 4. Boxplot for tenure
ax4 = plt.subplot(3, 3, 4)
sns.boxplot(data=df, x='Churn', y='tenure', ax=ax4, palette=colors)
ax4.set_title('Tenure - Quartile Analysis', fontsize=14, fontweight='bold')
ax4.set_xlabel('Churn')
ax4.set_ylabel('Tenure (months)')
ax4.set_xticklabels(['No Churn', 'Churn'])

# 5. Boxplot for MonthlyCharges
ax5 = plt.subplot(3, 3, 5)
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=ax5, palette=colors)
ax5.set_title('Monthly Charges - Quartile Analysis', fontsize=14, fontweight='bold')
ax5.set_xlabel('Churn')
ax5.set_ylabel('Monthly Charges ($)')
ax5.set_xticklabels(['No Churn', 'Churn'])

# 6. Boxplot for TotalCharges
ax6 = plt.subplot(3, 3, 6)
sns.boxplot(data=df, x='Churn', y='TotalCharges', ax=ax6, palette=colors)
ax6.set_title('Total Charges - Quartile Analysis', fontsize=14, fontweight='bold')
ax6.set_xlabel('Churn')
ax6.set_ylabel('Total Charges ($)')
ax6.set_xticklabels(['No Churn', 'Churn'])

plt.tight_layout()
plt.show()
```
```
# Cell 6: Exploratory Data Analysis - Part 2
# ====================================================
# Analysis of important categorical variables
fig2 = plt.figure(figsize=(20, 10))

# 1. Contract vs Churn
ax1 = plt.subplot(2, 3, 1)
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn_percent = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100

contract_churn_percent.plot(kind='bar', stacked=True, ax=ax1, color=['#3498db', '#e74c3c'])
ax1.set_title('Contract Type and Churn', fontsize=14, fontweight='bold')
ax1.set_xlabel('Contract Type')
ax1.set_ylabel('Percentage')
ax1.legend(['No Churn', 'Churn'], title='Churn Status')
ax1.tick_params(axis='x', rotation=45)

# Add percentages
for i, (idx, row) in enumerate(contract_churn_percent.iterrows()):
    cumulative = 0
    for j, val in enumerate(row):
        if val > 5:  # Only show for significant values
            ax1.text(i, cumulative + val/2, f'{val:.0f}%', 
                    ha='center', va='center', color='white', fontweight='bold')
        cumulative += val

# 2. InternetService vs Churn
ax2 = plt.subplot(2, 3, 2)
internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
internet_churn_percent = internet_churn.div(internet_churn.sum(axis=1), axis=0) * 100

internet_churn_percent.plot(kind='bar', stacked=True, ax=ax2, color=['#3498db', '#e74c3c'])
ax2.set_title('Internet Service and Churn', fontsize=14, fontweight='bold')
ax2.set_xlabel('Internet Service')
ax2.set_ylabel('Percentage')
ax2.legend(['No Churn', 'Churn'], title='Churn Status')
ax2.tick_params(axis='x', rotation=45)

# 3. PaymentMethod vs Churn
ax3 = plt.subplot(2, 3, 3)
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'])
payment_churn_percent = payment_churn.div(payment_churn.sum(axis=1), axis=0) * 100

payment_churn_percent.plot(kind='bar', stacked=True, ax=ax3, color=['#3498db', '#e74c3c'])
ax3.set_title('Payment Method and Churn', fontsize=14, fontweight='bold')
ax3.set_xlabel('Payment Method')
ax3.set_ylabel('Percentage')
ax3.legend(['No Churn', 'Churn'], title='Churn Status')
ax3.tick_params(axis='x', rotation=45)

# 4. Gender vs Churn
ax4 = plt.subplot(2, 3, 4)
gender_churn = pd.crosstab(df['gender'], df['Churn'])
gender_churn_percent = gender_churn.div(gender_churn.sum(axis=1), axis=0) * 100

gender_churn_percent.plot(kind='bar', stacked=True, ax=ax4, color=['#3498db', '#e74c3c'])
ax4.set_title('Gender and Churn', fontsize=14, fontweight='bold')
ax4.set_xlabel('Gender')
ax4.set_ylabel('Percentage')
ax4.legend(['No Churn', 'Churn'], title='Churn Status')
ax4.tick_params(axis='x', rotation=0)

# 5. SeniorCitizen vs Churn
ax5 = plt.subplot(2, 3, 5)
senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'])
senior_churn_percent = senior_churn.div(senior_churn.sum(axis=1), axis=0) * 100

senior_churn_percent.plot(kind='bar', stacked=True, ax=ax5, color=['#3498db', '#e74c3c'])
ax5.set_title('Senior Citizen and Churn', fontsize=14, fontweight='bold')
ax5.set_xlabel('Senior Citizen')
ax5.set_ylabel('Percentage')
ax5.legend(['No Churn', 'Churn'], title='Churn Status')
ax5.set_xticklabels(['No', 'Yes'], rotation=0)

# 6. Partner vs Churn
ax6 = plt.subplot(2, 3, 6)
partner_churn = pd.crosstab(df['Partner'], df['Churn'])
partner_churn_percent = partner_churn.div(partner_churn.sum(axis=1), axis=0) * 100

partner_churn_percent.plot(kind='bar', stacked=True, ax=ax6, color=['#3498db', '#e74c3c'])
ax6.set_title('Partner and Churn', fontsize=14, fontweight='bold')
ax6.set_xlabel('Has Partner')
ax6.set_ylabel('Percentage')
ax6.legend(['No Churn', 'Churn'], title='Churn Status')
ax6.set_xticklabels(['No', 'Yes'], rotation=0)

plt.tight_layout()
plt.show()

# Statistical insights
print("\n📊 Key Insights from EDA:")
print("=" * 50)
print("1. Customers with month-to-month contracts have much higher churn rates")
print("2. Customers without internet service (No) have the lowest churn rate")
print("3. Electronic Check payment method has the highest churn rate")
print("4. Senior citizens have higher churn rates")
print("5. Customers without partners have slightly higher churn rates")
print("6. Gender doesn't have significant impact on churn rate")
```
```
# Cell 7: Feature/Target Split & Data Preparation
# ====================================================
print("⚙️ Preparing data for modeling")
print("=" * 50)

# 1. Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# 2. Identify feature types
cat_features = X.select_dtypes(include=['object']).columns.tolist()
num_features = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n🔤 Categorical features ({len(cat_features)}):")
for feature in cat_features:
    unique_vals = X[feature].nunique()
    print(f"  - {feature}: {unique_vals} unique values")

print(f"\n🔢 Numerical features ({len(num_features)}):")
print(f"  - {', '.join(num_features)}")

# 3. Display sample data
print("\n🔍 Sample of categorical data:")
sample_cat = X[cat_features].head(3)
print(sample_cat)

print("\n🔢 Sample of numerical data:")
sample_num = X[num_features].head(3)
print(sample_num)
```
```
# Cell 8: Train/Test Split
# ====================================================
print("📊 Splitting data into training and testing sets")
print("=" * 50)

# Split data while preserving class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # Fixed for reproducibility
    stratify=y           # Preserve class ratios in split
)

print(f"📚 Training Set (Train):")
print(f"  - Samples: {X_train.shape[0]}")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - Class distribution: {dict(y_train.value_counts())}")
print(f"  - Percentages: {y_train.value_counts(normalize=True).values}")

print(f"\n🧪 Testing Set (Test):")
print(f"  - Samples: {X_test.shape[0]}")
print(f"  - Features: {X_test.shape[1]}")
print(f"  - Class distribution: {dict(y_test.value_counts())}")
print(f"  - Percentages: {y_test.value_counts(normalize=True).values}")

# Verify preservation of distribution
print(f"\n✅ Verifying class distribution preservation:")
print(f"  Full dataset: {df['Churn'].mean():.3f}")
print(f"  Train set: {y_train.mean():.3f}")
print(f"  Test set:  {y_test.mean():.3f}")

if abs(y_train.mean() - y_test.mean()) < 0.01:
    print("  ✅ Class distributions are well preserved")
else:
    print("  ⚠️ Significant difference in class distributions")
```
```
# Cell 9: Data Preparation for Logistic Regression
# ====================================================
print("🔧 Preparing data for Logistic Regression")
print("=" * 50)

# 1. One-Hot Encoding for categorical features (without Data Leakage)
print("1. Applying One-Hot Encoding:")
print("   📌 Note: Preventing Data Leakage with separate encoding")

# Train encoding on train set only
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
print(f"   Train after encoding: {X_train_encoded.shape}")

# Apply same encoding to test set
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align columns (some values might not exist in test)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
print(f"   Test after encoding and alignment: {X_test_encoded.shape}")

print(f"\n   Number of features after encoding: {X_train_encoded.shape[1]}")
print(f"   (Increased from {X_train.shape[1]} original features)")

# 2. Standard Scaling (normalization)
print("\n2. Applying Standard Scaling:")
print("   📌 Essential for Logistic Regression (scale-sensitive)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)  # fit only on train
X_test_scaled = scaler.transform(X_test_encoded)        # transform on test

print(f"   ✅ Data normalized successfully")
print(f"   Mean after scaling (train): {X_train_scaled.mean():.2f}")
print(f"   Std after scaling (train): {X_train_scaled.std():.2f}")

# 3. Display sample information
print("\n3. Sample of prepared data:")
print(f"   X_train_scaled shape: {X_train_scaled.shape}")
print(f"   X_test_scaled shape: {X_test_scaled.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   y_test shape: {y_test.shape}")

# Save feature names for later analysis
feature_names = X_train_encoded.columns.tolist()
print(f"\n   Feature names saved: {len(feature_names)} features")
```
```
# Cell 10: Baseline Model - Logistic Regression
# ====================================================
print("🎯 Baseline Model: Logistic Regression")
print("=" * 50)

# 1. Create and train model
print("1. Creating and training model...")
baseline_model = LogisticRegression(
    max_iter=1000,           # Increase iterations for convergence
    class_weight='balanced', # Handle class imbalance
    random_state=42,         # Reproducibility
    solver='lbfgs'          # Optimization algorithm
)

# Training timing
import time
start_time = time.time()
baseline_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"   ✅ Model trained successfully ({training_time:.2f} seconds)")
print(f"   Number of iterations performed: {baseline_model.n_iter_[0]}")

# 2. Make predictions
print("\n2. Making predictions...")
y_pred_base = baseline_model.predict(X_test_scaled)
y_proba_base = baseline_model.predict_proba(X_test_scaled)[:, 1]  # Positive class probability

print(f"   ✅ Predictions completed")
print(f"   y_pred shape: {y_pred_base.shape}")
print(f"   y_proba shape: {y_proba_base.shape}")

# 3. Evaluate model
print("\n3. Evaluating model performance:")
print("-" * 30)

accuracy = accuracy_score(y_test, y_pred_base)
roc_auc = roc_auc_score(y_test, y_proba_base)
f1 = f1_score(y_test, y_pred_base)

print(f"   📊 Accuracy:  {accuracy:.4f}")
print(f"   📈 ROC-AUC:   {roc_auc:.4f}")
print(f"   ⚖️ F1-Score:  {f1:.4f}")

# 4. Cross-Validation for validation
print("\n4. Validating with Cross-Validation...")
cv_scores = cross_val_score(
    baseline_model,
    X_train_scaled,
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1  # Use all cores
)

print(f"   ✅ Cross-Validation completed (5-fold)")
print(f"   📊 CV Scores: {cv_scores}")
print(f"   📈 Mean CV AUC: {cv_scores.mean():.4f}")
print(f"   📉 Std CV AUC:  {cv_scores.std():.4f}")

# 5. Coefficient analysis
print("\n5. Model coefficient analysis:")
print("-" * 30)

coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': baseline_model.coef_[0],
    'Abs_Coefficient': np.abs(baseline_model.coef_[0])
})

# Top 10 features with positive impact
print("\n   🔼 Top 10 features with positive impact on Churn:")
top_positive = coefficients.sort_values('Coefficient', ascending=False).head(10)
print(top_positive[['Feature', 'Coefficient']].to_string(index=False))

# Top 10 features with negative impact
print("\n   🔽 Top 10 features with negative impact on Churn:")
top_negative = coefficients.sort_values('Coefficient', ascending=True).head(10)
print(top_negative[['Feature', 'Coefficient']].to_string(index=False))

print("\n✅ Logistic Regression model executed successfully!")
```
```

