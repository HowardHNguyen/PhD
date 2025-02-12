import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/howardnguyen/Documents/Projects/Heart Disease Project/heart_attack_2022_400k.csv', sep=',')
print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())
# Fill missing values with mean values
# data.fillna(data.mean(), inplace=True)
data.dropna(inplace=True)
print(data.isnull().sum())
# print(data['CVD'].value_counts())

#############################################################
# label encode
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Sample data
df = pd.DataFrame(data)

# Encode categorical variables
label_encoder = LabelEncoder()

# Apply label encoding to ordinal categorical columns
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['PhysicalActivities'] = label_encoder.fit_transform(df['PhysicalActivities'])
df['HeartAttack'] = label_encoder.fit_transform(df['HeartAttack'])
df['Angina'] = label_encoder.fit_transform(df['Angina'])
df['Stroke'] = label_encoder.fit_transform(df['Stroke'])
df['Asthma'] = label_encoder.fit_transform(df['Asthma'])
df['SkinCancer'] = label_encoder.fit_transform(df['SkinCancer'])
df['COPD'] = label_encoder.fit_transform(df['COPD'])
df['DepressiveDisorder'] = label_encoder.fit_transform(df['DepressiveDisorder'])
df['KidneyDisease'] = label_encoder.fit_transform(df['KidneyDisease'])
df['Arthritis'] = label_encoder.fit_transform(df['Arthritis'])
df['Diabetes'] = label_encoder.fit_transform(df['Diabetes'])
df['DifficultyWalking'] = label_encoder.fit_transform(df['DifficultyWalking'])
df['SmokerStatus'] = label_encoder.fit_transform(df['SmokerStatus'])
df['Alcohol'] = label_encoder.fit_transform(df['Alcohol'])

# One-hot encode nominal categorical columns
df = pd.get_dummies(df, columns=['AgeCategory', 'Race', 'GeneralHealth'])

# Calculate the correlation matrix
# corr_matrix = df.corr()

"""# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(28, 18))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.show()"""
#############################################################
# Mapping Yes/No to 1/0
yes_no_map = {'Yes': 1, 'No': 0}
data['PhysicalActivities'] = data['PhysicalActivities'].map(yes_no_map)
data['HeartAttack'] = data['HeartAttack'].map(yes_no_map)
data['Angina'] = data['Angina'].map(yes_no_map)
data['Stroke'] = data['Stroke'].map(yes_no_map)
data['Asthma'] = data['Asthma'].map(yes_no_map)
data['SkinCancer'] = data['SkinCancer'].map(yes_no_map)
data['COPD'] = data['COPD'].map(yes_no_map)
data['DepressiveDisorder'] = data['DepressiveDisorder'].map(yes_no_map)
data['KidneyDisease'] = data['KidneyDisease'].map(yes_no_map)
data['Arthritis'] = data['Arthritis'].map(yes_no_map)
data['Diabetes'] = data['Diabetes'].map(yes_no_map)
data['DifficultyWalking'] = data['DifficultyWalking'].map(yes_no_map)
data['SmokerStatus'] = data['SmokerStatus'].map(yes_no_map)
data['Alcohol'] = data['Alcohol'].map(yes_no_map)

# Mapping Female/Male to 0/1
sex_map = {'Female': 0, 'Male': 1}
data['Sex'] = data['Sex'].map(sex_map)

# Mapping General Health categories (you can customize this mapping as needed)
gen_health_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Very good': 4, 'Excellent': 5}
data['GeneralHealth'] = data['GeneralHealth'].map(gen_health_map)

# Mapping Race categories (you can customize this mapping as needed)
race_map = {'White': 1, 'Black': 2, 'Hispanic': 3, 'Multiracial': 4,  'Other':5}
data['Race'] = data['Race'].map(race_map)

# AgeCategory mapping (based on the assumption that you might want a similar mapping)
age_category_map = {
    '18-24': 1, '25-29': 2, '30-34': 3, '35-39': 4,
    '40-44': 5, '45-49': 6, '50-54': 7, '55-59': 8,
    '60-64': 9, '65-69': 10, '70-74': 11, '75-79': 12, '80 or older': 13
}
data['AgeCategory'] = data['AgeCategory'].map(age_category_map)

# Display the updated DataFrame
# print(data)
#############################################################
# Calculate the correlation matrix
corr_matrix = data.corr()

# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(18, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.show()

#############################################################

# FEATURE IMPORTANCES by GAN Method
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
#data = pd.read_csv('/Users/howardnguyen/Documents/Projects/Heart Disease Project/heart_attack_2022_400k.csv', sep=',')
#data.dropna(inplace=True)

# Define features and target
X = data[['Sex', 'GeneralHealth', 'PhysicalActivities', 'SleepHours',
       'Angina', 'Stroke', 'Asthma', 'SkinCancer', 'COPD',
       'DepressiveDisorder', 'KidneyDisease', 'Arthritis', 'Diabetes',
       'DifficultyWalking', 'SmokerStatus', 'Race', 'AgeCategory', 'BMI',
       'Alcohol']]
y = data['HeartAttack']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# FEATURE IMPORTANCES
feature_importances_rf = rf.feature_importances_
features = X.columns
importance_df_rf = pd.DataFrame({'Feature': features, 'Importance': feature_importances_rf})

# Sort and display feature importances
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)
print("RF Feature Importances / Influential Predictors - SMOTE Balanced Dataset")
print(importance_df_rf)

# Plot the feature importances
plt.figure(figsize=(12, 5))
plt.barh(importance_df_rf['Feature'], importance_df_rf['Importance'])
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('RF Feature Importances / Influential Predictors - SMOTE Balanced Dataset')
plt.show()
##############################################################

# VISUALIZE SMOTE Synthetic Data Points
from collections import Counter

# Simulated counts
before_smote = Counter(y)
after_smote = Counter(y_resampled)

# Calculate synthetic points
synthetic_points = after_smote[1] - before_smote[1]

# Bar plot to visualize before and after SMOTE
categories = ['Before SMOTE (Minority)', 'Synthetic Points', 'After SMOTE (Minority)']
counts = [before_smote[1], synthetic_points, after_smote[1]]

plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['blue', 'orange', 'green'], alpha=0.7)
plt.title('Synthetic Points Generated via SMOTE')
plt.ylabel('Count')
plt.xlabel('Category')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate counts above the bars
for i, count in enumerate(counts):
    plt.text(i, count + 5000, f'{count:,}', ha='center', fontsize=10)

plt.show()

#######################################################

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

# Separate the minority and majority classes
X_minority = X[y == 1]  # Features only
X_majority = X[y == 0]
y_majority = y[y == 0]

# Apply SMOTE to generate synthetic data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Prepare real_data and synthetic_data for visualization
real_data = pd.concat([X_majority, X_minority], axis=0)  # Real data
synthetic_data = X_resampled[len(real_data):]  # Only newly generated synthetic data

# Scale features
scaler = StandardScaler()
real_data_scaled = scaler.fit_transform(real_data)
synthetic_data_scaled = scaler.transform(synthetic_data)

# Combine real and synthetic data for PCA
combined_data = np.vstack([real_data_scaled, synthetic_data_scaled])
labels = np.array([0] * len(real_data_scaled) + [1] * len(synthetic_data_scaled))  # 0 for real, 1 for synthetic

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)

# Separate the reduced data
real_data_pca = pca_result[labels == 0]
synthetic_data_pca = pca_result[labels == 1]

# Plot PCA visualization
plt.figure(figsize=(8, 6))
plt.scatter(real_data_pca[:, 0], real_data_pca[:, 1], alpha=0.7, label="Real Data", color='blue')
plt.scatter(synthetic_data_pca[:, 0], synthetic_data_pca[:, 1], alpha=0.7, label="Synthetic Data", color='orange')
plt.title("PCA Visualization of Real vs. Synthetic Data (SMOTE)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(alpha=0.5)
plt.show()

#######################################################


#######################################################
# Based ML Models
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Logistic Regression with L2 regularization
# lr = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Evaluation
print("Logistic Regression on dataset")
print(classification_report(y_test, y_pred_lr))
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
print(f'ROC AUC: {roc_auc_lr:.2f}')

# Plot the ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
"""
plt.figure()
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
"""
# Support Vector Machine Model
from sklearn.svm import SVC
"""
# Support Vector Machine with reduced complexity
svm = SVC(C=1.0, kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Predictions
y_pred_svm = svm.predict(X_test)
y_proba_svm = svm.predict_proba(X_test)[:, 1]

# Evaluation
print("Support Vector Machine on dataset")
print(classification_report(y_test, y_pred_svm))
roc_auc_svm = roc_auc_score(y_test, y_proba_svm)
print(f'ROC AUC: {roc_auc_svm:.2f}')

# Plot the ROC curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
"""

"""plt.figure()
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'Support Vector Machine (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()"""
######################################################################

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Random Forest: base model with further reduced complexity
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluation
print("Random Forest / 400k dataset")
print(classification_report(y_test, y_pred_rf))
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f'ROC AUC: {roc_auc_rf:.2f}')
######################################################################

# Gradient Boosting Machine Model
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting Machine
gbm = GradientBoostingClassifier(max_depth=3, n_estimators=30, min_samples_leaf=5, random_state=42)
gbm.fit(X_train, y_train)

# Predictions
y_pred_gbm = gbm.predict(X_test)
y_proba_gbm = gbm.predict_proba(X_test)[:, 1]

# Evaluation
print("Gradient Boosting Machine / 400k dataset")
print(classification_report(y_test, y_pred_gbm))
roc_auc_gbm = roc_auc_score(y_test, y_proba_gbm)
print(f'ROC AUC: {roc_auc_gbm:.2f}')
######################################################################

# Extreme Gradient Boosting Machine Model
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# XGBoost Classifier
xgbm = XGBClassifier(max_depth=3, min_child_weight=10, subsample=0.7, colsample_bytree=0.7, random_state=42)
xgbm.fit(X_train, y_train)

# Predictions
y_pred_xgbm = xgbm.predict(X_test)
y_proba_xgbm = xgbm.predict_proba(X_test)[:, 1]

# Evaluation
print("XGBoost Classifier / 400k dataset")
print(classification_report(y_test, y_pred_xgbm))
roc_auc_xgbm = roc_auc_score(y_test, y_proba_xgbm)
print(f'ROC AUC: {roc_auc_xgbm:.2f}')

######################################################################

# Convolutional Neural Network (CNN)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# Reshape data to add timesteps dimension (e.g., timesteps=1)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# CNN Model
# Define input_dim based on your dataset
input_dim = X_train.shape[1]

cnn_model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predictions
y_pred_cnn = cnn_model.predict(X_test_reshaped).ravel()
y_pred_cnn_class = (y_pred_cnn > 0.5).astype(int)

# Evaluation
print("CNN / 400k dataset")
print(classification_report(y_test, y_pred_cnn_class))
roc_auc_cnn = roc_auc_score(y_test, y_pred_cnn)
print(f'ROC AUC: {roc_auc_cnn:.2f}')

######################################################################
"""
# GRU Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define input_dim based on your dataset
input_dim = X_train.shape[1]

# Simple Neural Network with reduced complexity
nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predictions
y_pred_nn = nn_model.predict(X_test).ravel()
y_pred_nn_class = (y_pred_nn > 0.5).astype(int)

# Evaluation
print("Simple Neural Network / 400k dataset")
print(classification_report(y_test, y_pred_nn_class))
roc_auc_nn = roc_auc_score(y_test, y_pred_nn)
print(f'ROC AUC: {roc_auc_nn:.2f}')

# Plot the ROC curve
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nn)
plt.figure()
plt.plot(fpr_nn, tpr_nn, color='blue', lw=2, label=f'Simple Neural Network (AUC = {roc_auc_nn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()"""
###########################################################################

# CNN with GRU Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

# Define input_dim and timesteps based on your dataset
timesteps = X_train_reshaped.shape[1]
input_dim = X_train_reshaped.shape[2]

# CNN with GRU
cnn_model_gru = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(timesteps, input_dim)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    GRU(32, return_sequences=True),
    Dropout(0.5),
    GRU(16),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model_gru.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predictions
y_pred_cnn_gru = cnn_model_gru.predict(X_test_reshaped).ravel()
y_pred_cnn_gru_class = (y_pred_cnn_gru > 0.5).astype(int)

# Evaluation
print("CNN with GRU / 400k dataset")
print(classification_report(y_test, y_pred_cnn_gru_class))
roc_auc_cnn_gru = roc_auc_score(y_test, y_pred_cnn_gru)
print(f'ROC AUC: {roc_auc_cnn_gru:.2f}')

# Plot the ROC curve
fpr_cnn_gru, tpr_cnn_gru, _ = roc_curve(y_test, y_pred_cnn_gru)
"""plt.figure()
plt.plot(fpr_cnn_gru, tpr_cnn_gru, color='blue', lw=2, label=f'CNN with GRU (AUC = {roc_auc_cnn_gru:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve on dataset Framingham')
plt.legend(loc="lower right")
plt.show()"""
######################################################################

# STACKING MODELS RF + xGBM + CNN
# Import necessary libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Check if X_train and X_test are already NumPy arrays
if not isinstance(X_train, np.ndarray):
    X_train_np = X_train.to_numpy()
else:
    X_train_np = X_train

if not isinstance(X_test, np.ndarray):
    X_test_np = X_test.to_numpy()
else:
    X_test_np = X_test

# Define and train traditional machine learning models
rf = RandomForestClassifier(max_depth=3, n_estimators=30, min_samples_leaf=5, random_state=42)
xgbm = XGBClassifier(max_depth=3, min_child_weight=10, subsample=0.7, colsample_bytree=0.7, random_state=42)

rf.fit(X_train_np, y_train)
xgbm.fit(X_train_np, y_train)

# Define and train a CNN model
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

X_train_cnn = X_train_np.reshape(X_train_np.shape[0], X_train_np.shape[1], 1)
X_test_cnn = X_test_np.reshape(X_test_np.shape[0], X_test_np.shape[1], 1)

cnn_model = create_cnn_model((X_train_np.shape[1], 1))
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Generate predictions for training data
rf_train_pred = rf.predict_proba(X_train_np)[:, 1]
xgbm_train_pred = xgbm.predict_proba(X_train_np)[:, 1]

cnn_train_pred = cnn_model.predict(X_train_cnn).ravel()

# Combine predictions into a single training set for the meta-learner
stacked_train_pred = np.column_stack((rf_train_pred, xgbm_train_pred, cnn_train_pred))

# Generate predictions for test data
rf_test_pred = rf.predict_proba(X_test_np)[:, 1]
xgbm_test_pred = xgbm.predict_proba(X_test_np)[:, 1]

cnn_test_pred = cnn_model.predict(X_test_cnn).ravel()

# Combine predictions into a single test set for the meta-learner
stacked_test_pred = np.column_stack((rf_test_pred, xgbm_test_pred, cnn_test_pred))

# Train the meta-learner on the stacked predictions
meta_model = LogisticRegression()
meta_model.fit(stacked_train_pred, y_train)

# Evaluate the stacking ensemble
y_pred_stack = meta_model.predict(stacked_test_pred)
y_proba_stack = meta_model.predict_proba(stacked_test_pred)[:, 1]

# Print evaluation metrics
print("Stacking Ensemble with RF + xGBM + CNN on 400K dataset")
print(classification_report(y_test, y_pred_stack))
roc_auc_stack = roc_auc_score(y_test, y_proba_stack)
print(f'ROC AUC with RF + xGBM + CNN on 400K dataset: {roc_auc_stack:.2f}')

# Plot the ROC curve
fpr_stacking_ml, tpr_stacking_ml, _ = roc_curve(y_test, y_proba_stack)
roc_auc_stacking_ml = roc_auc_stack  # Use the correct AUC value
"""plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Stacking Ensemble (AUC = {roc_auc_stack:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve on 400K dataset')
plt.legend(loc="lower right")
plt.show()"""
######################################################################

# GENERATIVE AI MODEL with SMOTE
# Step 1: Data Preparation and Balancing Using SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from collections import Counter
import pandas as pd
import numpy as np

# Load dataset - already loaded in 'data'
# Define features and target
X = data[['Sex', 'GeneralHealth', 'PhysicalActivities', 'SleepHours',
       'Angina', 'Stroke', 'Asthma', 'SkinCancer', 'COPD',
       'DepressiveDisorder', 'KidneyDisease', 'Arthritis', 'Diabetes',
       'DifficultyWalking', 'SmokerStatus', 'Race', 'AgeCategory', 'BMI',
       'Alcohol']]
y = data['HeartAttack']

print("Class distribution before balancing:", Counter(y))

# Step 2: Apply SMOTE to generate synthetic data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after SMOTE balancing:", Counter(y_resampled))

# Step 3: Model Training and Evaluation

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the RandomForestClassifier model
genai_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
genai_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = genai_model.predict(X_test)
y_pred_proba = genai_model.predict_proba(X_test)[:, 1]

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc_genai = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred)

# Output the evaluation metrics
print("Accuracy / Gen AI / 400k dataset:", accuracy)
print("ROC AUC / Gen AI / 400k dataset:", roc_auc_genai)
print("Classification Report / Heart Attack / 400K dataset:\n", classification_rep)

# Plot the ROC Curve
fpr_genai, tpr_genai, _ = roc_curve(y_test, y_pred_proba)

"""plt.figure()
plt.plot(fpr_genai, tpr_genai, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_genai:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve / Heart Attack / 400K dataset (SMOTE)')
plt.legend(loc="lower right")
plt.show()"""
############################################################

# STACKING Gen AI + RF + xGBM + CNN with SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import numpy as np
import pandas as pd

# Step 1: Load and preprocess the dataset
X = data[['Sex', 'GeneralHealth', 'PhysicalActivities', 'SleepHours',
       'Angina', 'Stroke', 'Asthma', 'SkinCancer', 'COPD',
       'DepressiveDisorder', 'KidneyDisease', 'Arthritis', 'Diabetes',
       'DifficultyWalking', 'SmokerStatus', 'Race', 'AgeCategory', 'BMI',
       'Alcohol']]
y = data['HeartAttack']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to generate synthetic data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 3: Train models (Random Forest, GBM, CNN)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=42)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# CNN setup
cnn_model = Sequential([
    Conv1D(16, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Stacking predictions
meta_train_rf = rf_model.predict_proba(X_train)[:, 1]
meta_train_xgb = xgb_model.predict_proba(X_train)[:, 1]
meta_train_cnn = cnn_model.predict(X_train_cnn).ravel()
X_meta_train = np.column_stack([meta_train_rf, meta_train_xgb, meta_train_cnn])

meta_model = LogisticRegression()
meta_model.fit(X_meta_train, y_train)

# Evaluate the stacking model
meta_test_rf = rf_model.predict_proba(X_test)[:, 1]
meta_test_xgb = xgb_model.predict_proba(X_test)[:, 1]
meta_test_cnn = cnn_model.predict(X_test_cnn).ravel()
X_meta_test = np.column_stack([meta_test_rf, meta_test_xgb, meta_test_cnn])
meta_predictions = meta_model.predict_proba(X_meta_test)[:, 1]
meta_class_predictions = (meta_predictions > 0.5).astype(int)

print("Classification Report / Stacking Gen AI-SMOTE / 400k dataset:\n", classification_report(y_test, meta_class_predictions))
roc_auc_stacking_genai_SMOTE = roc_auc_score(y_test, meta_predictions)
print(f"ROC AUC: {roc_auc_stacking_genai_SMOTE:.4f}")

# Plot ROC Curve
fpr_stacking_genai, tpr_stacking_genai, _ = roc_curve(y_test, meta_predictions)

"""plt.figure()
plt.plot(fpr_stacking_genai, tpr_stacking_genai, label=f'Stacking Model (AUC = {roc_auc_stacking_genai_SMOTE:.2f})', color='blue', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve / Stacking Gen AI-SMOTE / 400k dataset')
plt.legend(loc='lower right')
plt.show()"""
##################################################################

from sklearn.metrics import roc_curve, roc_auc_score

# Calculate ROC curve and AUC for each updated model
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_proba_gbm)
roc_auc_gbm = roc_auc_score(y_test, y_proba_gbm)

fpr_xgbm, tpr_xgbm, _ = roc_curve(y_test, y_proba_xgbm)
roc_auc_xgbm = roc_auc_score(y_test, y_proba_xgbm)

fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_pred_cnn)
roc_auc_cnn = roc_auc_score(y_test, y_pred_cnn)

fpr_cnn_gru, tpr_cnn_gru, _ = roc_curve(y_test, y_pred_cnn_gru)
roc_auc_cnn_gru = roc_auc_score(y_test, y_pred_cnn_gru)

# ✅ Ensure correct variable names
fpr_stacking_ml, tpr_stacking_ml, _ = roc_curve(y_test, y_proba_stack)
roc_auc_stacking_ml = roc_auc_score(y_test, y_proba_stack)

fpr_genai, tpr_genai, _ = roc_curve(y_test, y_pred_proba)
roc_auc_genai = roc_auc_score(y_test, y_pred_proba)

fpr_stacking_genai, tpr_stacking_genai, _ = roc_curve(y_test, meta_predictions)
roc_auc_stacking_genai_GAN = roc_auc_score(y_test, meta_predictions)

# ✅ Plot all ROC curves on the same plot
def plot_roc_curves():
    plt.figure(figsize=(10, 7))

    # Plot ROC Curves for different models
    plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, color='gray', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot(fpr_gbm, tpr_gbm, color='purple', lw=2, label=f'Gradient Boosting (AUC = {roc_auc_gbm:.2f})')
    plt.plot(fpr_xgbm, tpr_xgbm, color='darkred', lw=2, label=f'xGradient Boosting (AUC = {roc_auc_xgbm:.2f})')
    plt.plot(fpr_cnn, tpr_cnn, color='brown', lw=2, label=f'Convolutional Neural Network (AUC = {roc_auc_cnn:.2f})')
    plt.plot(fpr_cnn_gru, tpr_cnn_gru, color='cyan', lw=2, label=f'CNN with GRU (AUC = {roc_auc_cnn_gru:.2f})')

    # ✅ Corrected FPR, TPR, and AUC values
    plt.plot(fpr_genai, tpr_genai, color='red', lw=2, label=f'Stand-Alone GenAI Model (AUC = {roc_auc_genai:.3f})')
    plt.plot(fpr_stacking_ml, tpr_stacking_ml, color='yellow', lw=2, label=f'Stacking ML Models (AUC = {roc_auc_stacking_ml:.3f})')
    plt.plot(fpr_stacking_genai, tpr_stacking_genai, color='green', lw=2, label=f'Stacking GenAI-GAN with RF+xGBM+CNN Models (AUC = {roc_auc_stacking_genai_GAN:.3f})')

    # Reference diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Plot settings
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve / All Models / 400k dataset')
    plt.legend(loc="lower right")

    # Show the plot
    plt.show()

# Call the function
print("Starting function execution...")
plot_roc_curves()
print("Function execution completed. Awesome!")