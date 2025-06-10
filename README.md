# 🔋 Solar Panel Efficiency Prediction

**Challenge:** Zelestra X AWS ML Ascend Challenge – 2nd Edition  
**Platform:** [HackerEarth Challenge Page](https://www.hackerearth.com/challenges/competitive/zelestra-x-aws-ml-ascend-challenge-second-edition/instructions/)  
**Objective:** Predict solar panel efficiency based on sensor and metadata.

---

## 📦 Dataset

The dataset was provided as part of the HackerEarth competition. It includes real-world features such as:

- Environmental readings (e.g., temperature, irradiance, humidity, wind speed)
- Panel-specific attributes (e.g., age, soiling ratio)
- Electrical readings (e.g., voltage, current)
- Maintenance history
- Efficiency (target variable)

[Dataset Access Link](https://www.hackerearth.com/challenges/competitive/zelestra-x-aws-ml-ascend-challenge-second-edition/instructions/)

---

## 🧠 Approach

### 1. 🧹 Data Cleaning & Preprocessing

- Checked for **missing values**, **duplicates**, and **anomalies** using `df.describe()` and visual inspections.
- **Clipped** numerical values (e.g., irradiance, temperature) based on realistic physical ranges from domain literature.
- **Removed rows** where `efficiency < 0.14` due to inconsistency between features and target (likely sensor errors).

### 2. 🧮 Missing Value Imputation

- **Numerical columns:** Used `IterativeImputer` (outperformed mean, median, and KNN).
- **Categorical columns:** Filled missing values with `"nan"` as a string placeholder.

### 3. ⚙️ Feature Engineering

Tried several domain-inspired engineered features:

- `power = voltage * current`
- `temp_penalty = max(0, temperature - 25) * 0.005`  (models temperature-related efficiency loss)
- `temp_diff = module_temperature - temperature`

🔍 **Result:** These features **did not improve** model performance. Literature also showed no consistent gains from engineered features when core sensors are already present.

### 4. 📊 Exploratory Data Analysis (EDA)

- Scatter plots to visualize trends between each feature and `efficiency`.
- **KDE plots** to study feature distribution in low-efficiency (<0.3) vs. high-efficiency (>0.8) ranges.
- **Correlation heatmap** to study multicollinearity between features and with the target.

### 5. 🔍 Feature Selection

- Tested both **mutual information gain** and **correlation-based filtering**.
- Final method: Retained features with **correlation > 0.01** with target.

**Selected Features:**

```python
['temperature', 'irradiance', 'humidity', 'panel_age',
 'maintenance_count', 'soiling_ratio', 'module_temperature',
 'voltage', 'current']
```


## 🤖 Modeling Strategy

### ✅ Models Tried

#### 🧠 Machine Learning
- Random Forest (RF)  
- XGBoost  
- CatBoost  
- Support Vector Regression (SVR)  

#### 🧪 Deep Learning
- Artificial Neural Network (ANN)  
- Extreme Learning Machine (ELM)  

---

### 🏆 Final Model

- **Ensemble and stacking** of RF, XGBoost, and CatBoost gave the **best performance**.
- Deep learning models **underperformed** due to limited dataset size and noise.

---

### 🔧 Training Details

- **Train/Test Split:** 90% training, 10% validation  
- **Scaling:** `RobustScaler` worked best due to presence of outliers

---

### 📉 Failed / Discarded Approaches

- Replacing `'unknown'`, `'badval'`, `'error'` in numeric columns and imputing — **worsened performance**
- Imputation with **mean**, **median**, or **KNN** — performed **worse than IterativeImputer**
- Deep learning models (**ANN**, **ELM**) — consistently **underperformed**
- Group-based imputation using **string IDs** — led to **overfitting**
- **Residual-based correction** (regressing residuals) — **made performance worse**
- Domain-specific engineered features — **no measurable gains**

