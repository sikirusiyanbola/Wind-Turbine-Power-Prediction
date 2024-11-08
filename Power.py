#Importing the needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Loading the dataset for model training
data = pd.read_csv('C:/Users/user/Desktop/MSc Folder/Wind_Turbine_Filtered.csv')

X = data.drop(columns=['LV ActivePower', 'Date/Time'])
y = data['LV ActivePower']

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Applying feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining hyperparameter grid for tuning
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Performing hyperparameter tuning using GridSearchCV
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Obtaining best parameters from tuning
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Model training using the best parameters
best_model = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error on test set: {mae}')

# Computing the metrics
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² (Goodness of Fit): {r2:.2f}")

# Converting the predictions to a DataFrame for better visualization
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())

# Plotting feature importances (Sensitivity analysis)
importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 3))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Importance')
plt.savefig('C:/Users/user/Desktop/Seasonal/image21', dpi=300)
plt.show()

# Visualizing actual values vs predicted values
plt.figure(figsize=(10, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Active Power')
plt.savefig('C:/Users/user/Desktop/Seasonal/image22', dpi=300)
plt.show()

# Plotting the Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.savefig('C:/Users/user/Desktop/Seasonal/image23', dpi=300)
plt.show()

# Computing the Average prediction
average_predicted_lv_active_power = np.mean(y_pred)
print(f'Average Predicted LV Active Power: {average_predicted_lv_active_power}')

# Calculating the average, highest, and lowest predicted Power
average_predicted_lv_activepower = np.mean(y_pred)
highest_predicted_lv_activepower = np.max(y_pred)
lowest_predicted_lv_activepower = np.min(y_pred)

print(f'Average Predicted LV Active Power: {average_predicted_lv_activepower:.2f}')
print(f'Highest Predicted LV Active Power: {highest_predicted_lv_activepower:.2f}')
print(f'Lowest Predicted LV Active Power: {lowest_predicted_lv_activepower:.2f}')

# Identifying anomalies using Isolation Forest and Local Outlier

# Fitting the model
iso_forest = IsolationForest(contamination=0.01)  # 1% contamination rate
data['anomaly_iso'] = iso_forest.fit_predict(data[['LV ActivePower']])

# Identifying the anomalies
anomalies_iso = data[data['anomaly_iso'] == -1]
print(f"Isolation Forest Anomalies:\n{anomalies_iso}")

# Removing the anomaly_iso column after using it
data.drop(columns=['anomaly_iso'], inplace=True)

# Fitting the model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)  # 1% contamination rate
data['anomaly_lof'] = lof.fit_predict(data[['LV ActivePower']])

# Identifying the anomalies
anomalies_lof = data[data['anomaly_lof'] == -1]
print(f"Local Outlier Factor Anomalies:\n{anomalies_lof}")

# Removing the anomaly_lof column after using it
data.drop(columns=['anomaly_lof'], inplace=True)

# Plotting the original data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['LV ActivePower'], label='ActivePower')

# Plotting anomalies detected by Isolation Forest
plt.scatter(anomalies_iso.index, anomalies_iso['LV ActivePower'], color='purple', label='Anomalies (Isolation Forest)', s=50)

# Plotting anomalies detected by LOF
plt.scatter(anomalies_lof.index, anomalies_lof['LV ActivePower'], color='orange', label='Anomalies (LOF)', s=50)

plt.xlabel('Date/Time')
plt.ylabel('Active Power')
plt.title('Active Power with Anomalies')
plt.legend()
plt.savefig('C:/Users/user/Desktop/Seasonal/image24', dpi=300)
plt.show()

