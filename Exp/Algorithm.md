# الگوریتم GradientBoosting (Catbooost) 

## 1️⃣ Boosting یعنی چی؟

درکل Boosting یک تکنیک Ensemble هست.

ایده‌ی اصلی:
مدل‌های ضعیف (weak learners) رو پشت‌سرهم می‌سازیم و هر مدل جدید سعی می‌کنه اشتباهات مدل قبلی رو اصلاح کنه.

مثلاً:

مدل اول پیش‌بینی می‌کنه

خطاها رو حساب می‌کنیم

مدل دوم روی خطاها تمرکز می‌کنه

مدل سوم روی خطاهای باقی‌مونده تمرکز می‌کنه

در نهایت مجموع وزن‌دار همه مدل‌ها خروجی نهایی رو می‌سازه

در اغلب موارد، weak learner ها درخت تصمیم کم‌عمق هستن.


## 2️⃣ Gradient یعنی چی؟

در یادگیری ماشین ما یک تابع خطا (Loss Function) داریم:

L(y,F(x))

هدف ما کمینه کردن این تابع است.

پس Gradient یعنی:
جهت بیشترین افزایش تابع.

ما برای کمینه‌سازی از منفی گرادیان استفاده می‌کنیم (مشابه Gradient Descent).

در Gradient Boosting:
هر مدل جدید تقریباً منفی گرادیان تابع خطا نسبت به پیش‌بینی فعلی رو یاد می‌گیره.


## 3️⃣ Gradient Boosting چیست؟

ترکیب دو مفهوم بالا:

مدل‌ها پشت سر هم ساخته می‌شوند (Boosting)

هر مدل جدید گرادیان خطا را تقریب می‌زند

مدل نهایی:

$$
F(x) = \sum_{m=1}^{M} \gamma_m h_m(x)
$$


که در این فرمول:

+ متغیر $h_m(x)$ یعنی درخت m ام
+ متغیر $𝛾_{m}$ وزن آن
+ متغیر M تعداد درخت ها
## 4️⃣ الگوریتم رسمی Gradient Boosting (فرمولی)

مرحله 1: مقداردهی اولیه

$$
F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, \gamma)
$$

مثلاً در رگرسیون:
میانگین y

مرحله 2: برای هر مرحله m:

1️⃣ محاسبه pseudo-residual:

$$
r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]
$$

2️⃣ آموزش درخت روی residual ها:

$r_{im}$ ≈ $h_m(x)$

3️⃣ پیدا کردن وزن بهینه:


$$
\gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L\left(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)\right)
$$

4️⃣ آپدیت مدل:

$$
F_m(x) = F_{m-1}(x) + \eta \gamma_m h_m(x)
$$

که:
η = learning rate

## 5️⃣ حالا CatBoost چیست؟

CatBoost
کتابخانه‌ای از شرکت Yandex است.

مخفف:
Categorical Boosting

مشکل اصلی که حل می‌کند:

+ داده‌های categorical

+ مشکل overfitting

+ مشکل target leakage

  ## 6️⃣ تفاوت اصلی CatBoost با Gradient Boosting معمولی
1️⃣ Ordered Boosting

در Gradient Boosting معمولی:
وقتی residual حساب می‌کنیم، از کل داده استفاده می‌کنیم → باعث leakage می‌شود.

CatBoost از تکنیک Ordered Boosting استفاده می‌کند:
هر نمونه فقط از داده‌های قبلی برای محاسبه گرادیان استفاده می‌کند.

این کار bias را کاهش می‌دهد.

2️⃣ مدیریت حرفه‌ای داده‌های categorical

به جای:
One-hot encoding

CatBoost از:

Target Encoding هوشمند و بدون leakage

فرمول ساده:

$$
\hat{x}_i = \frac{\sum y_{\text{prev}} + aP}{n_{\text{prev}} + a}
$$

که:

+ فقط نمونه‌های قبلی استفاده می‌شوند

+ متغیر a پارامتر regularization

+ متغیر P میانگین کلی target

  ## 7️⃣ مراحل اجرای CatBoost

1.تعیین Loss Function

2.مقداردهی اولیه مدل

3.Ordered Gradient Calculation

4.ساخت درخت متقارن (Oblivious Tree)

5.محاسبه leaf values

6.آپدیت مدل

7.تکرار

## 8️⃣ درخت‌های متقارن (Oblivious Trees)

CatBoost از درخت‌های خاصی استفاده می‌کند:

در هر سطح:
همه گره‌ها از یک feature و threshold استفاده می‌کنند.

مزایا:

سریع‌تر

کمتر overfit

GPU-friendly

پیش‌بینی سریع

# 9️⃣ مزایا

✔ مدیریت عالی categorical
✔ overfitting کمتر
✔ نیاز کمتر به preprocessing
✔ عملکرد قوی روی داده‌های کوچک
✔ مقاوم به leakage

#🔟 معایب

✖ کندتر از LightGBM در برخی موارد
✖ مصرف حافظه بیشتر
✖ در دیتاست‌های خیلی بزرگ گاهی کندتر

# 1️⃣1️⃣ کاربردها

+ Credit scoring

+ Fraud detection

+ Ranking

+ Recommender systems

+ Tabular data competitions

در مسابقات Kaggle بسیار محبوب است.
