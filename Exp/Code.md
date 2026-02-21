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
