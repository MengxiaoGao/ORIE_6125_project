import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

# 1. Read data (Replace with your actual file path if needed)
# cd /Users/gaomengxiao/Desktop
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Preview data and basic statistics
print(data.head())
print(data.describe())
print(data.corr())

# 2. Logical variable counting (using 'anaemia' as an example)
print("Anaemia=0 & DEATH_EVENT=0:", len(data[(data['anaemia'] == 0) & (data['DEATH_EVENT'] == 0)]))
print("Anaemia=0 & DEATH_EVENT=1:", len(data[(data['anaemia'] == 0) & (data['DEATH_EVENT'] == 1)]))

# Statistics for continuous variables (subset where DEATH_EVENT == 0)
subset1 = data[data['DEATH_EVENT'] == 0]
print(subset1.describe())

# 3. Plotting (Scatter plot matrix)
# Extract the 7 specified columns
cols_to_plot = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                'platelets', 'serum_creatinine', 'serum_sodium', 'time']
c1 = data[cols_to_plot].copy()
c1['Group'] = '1' # Equivalent to c1$Group = 1 in R

# Draw pairplot using seaborn
sns.pairplot(c1, hue='Group', palette="husl")
plt.show()

# 4. Data Standardization (Z-score scaling)
cols_to_scale = ['high_blood_pressure', 'anaemia', 'age', 'creatinine_phosphokinase', 
                 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
scaled_cols = [col + '_scaled' for col in cols_to_scale]

scaler = StandardScaler()
# Note: R's scale() uses unbiased estimation (ddof=1) by default, 
# sklearn uses biased estimation, but it doesn't affect the overall distribution.
data[scaled_cols] = scaler.fit_transform(data[cols_to_scale])

# 5. Principal Component Analysis (PCA)
# (1) Manual calculation of covariance matrix and eigenvalues
data1 = data[scaled_cols].values
n = data1.shape[0]
mx = np.eye(n) - np.ones((n, n)) / n
covA = data1.T @ mx @ data1 / (n - 1)
eigen_values, eigen_vectors = np.linalg.eig(covA)

# Draw Scree Plot
plt.figure()
plt.plot(range(1, 9), eigen_values, marker='o', color="#458B74")
plt.title("Scree Plot")
plt.xlabel("Component Number")
plt.ylabel("Eigenvalues")
plt.show()

# (2) Using built-in PCA function
pca = PCA(n_components=2)
data_pca_x = pca.fit_transform(data[scaled_cols])
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Loadings (Rotation):\n", pca.components_)

# 6. Linear Discriminant Analysis (LDA) & Leave-One-Out Cross Validation
lda = LinearDiscriminantAnalysis()
loo = LeaveOneOut()
y_true = data['DEATH_EVENT']

# L1: Using all scaled features
pred_L1 = cross_val_predict(lda, data[scaled_cols], y_true, cv=loo)

# L2: Using PCA features
pred_L2 = cross_val_predict(lda, data_pca_x, y_true, cv=loo)

# L3: Using 4 selected features (anaemia_scaled, age_scaled, cpk_scaled, platelets_scaled)
subset_cols = ['anaemia_scaled', 'age_scaled', 'creatinine_phosphokinase_scaled', 'platelets_scaled']
pred_L3 = cross_val_predict(lda, data[subset_cols], y_true, cv=loo)

# Confusion matrix and visualization function
def plot_confusion(y_t, y_p, title):
    cm = confusion_matrix(y_t, y_p)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion(y_true, pred_L1, "L1: All Scaled Features")
plot_confusion(y_true, pred_L2, "L2: PCA Features")
plot_confusion(y_true, pred_L3, "L3: Selected 4 Features")

# 7. Regression Analysis 
# Note: Using sm.OLS here to replicate the R `glm` without `family=binomial` (Linear Probability Model).
# To perform true Logistic Regression, replace sm.OLS with sm.Logit.
X_log = sm.add_constant(data[scaled_cols]) # Add constant term (intercept)
model_log = sm.OLS(y_true, X_log).fit()
print(model_log.summary())

# 8. Plotting stepwise regression results
# Calculate predicted scores Y1 and Y2 based on the hardcoded coefficients from your R script
Y1 = 0.321070 + 0.034980*data['high_blood_pressure_scaled'] + 0.029217*data['anaemia_scaled'] \
     + 0.103957*data['age_scaled'] + 0.045315*data['creatinine_phosphokinase_scaled'] \
     - 0.121507*data['ejection_fraction_scaled'] - 0.002756*data['platelets_scaled'] \
     + 0.109629*data['serum_creatinine_scaled'] - 0.049458*data['serum_sodium_scaled']

Y2 = 0.321070 + 0.10757*data['age_scaled'] - 0.13074*data['ejection_fraction_scaled'] \
     + 0.11902*data['serum_creatinine_scaled']

# Map colors based on DEATH_EVENT
color_map = {0: "#FF00FF", 1: "#000080"}
colors = data['DEATH_EVENT'].map(color_map)

# Plotting the classification effects of the two regressions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(range(len(Y1)), Y1, c=colors, alpha=0.7)
axes[0].set_title('Y1: Full Model Predictions')
axes[0].set_ylabel('Predicted Score')

axes[1].scatter(range(len(Y2)), Y2, c=colors, alpha=0.7)
axes[1].set_title('Y2: Stepwise Model Predictions')
axes[1].set_ylabel('Predicted Score')

plt.tight_layout()
plt.show()