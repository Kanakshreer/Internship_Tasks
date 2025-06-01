import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame

# Use all features except target
features = data.columns.drop('MedHouseVal')
X = data[features]
y = data['MedHouseVal']

# Handle missing values (if any)
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Hyperparameter tuning for Decision Tree
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
}

dt = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best Decision Tree model
dt_model = grid_search.best_estimator_
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)

print(f"Linear Regression MSE: {mse_lr:.3f}")
print(f"Tuned Decision Tree Regression MSE: {mse_dt:.3f}")
print(f"Best Decision Tree Params: {grid_search.best_params_}")

# Save the best model
if mse_lr < mse_dt:
    best_model = lr_model
    print("Saving Linear Regression model...")
else:
    best_model = dt_model
    print("Saving Tuned Decision Tree model...")

joblib.dump(best_model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
