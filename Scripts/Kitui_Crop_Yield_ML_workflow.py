import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import optuna
import seaborn as sns

!pip install catboost
!pip install optuna


original_file = 'Kitui_Crop_Yield_Yearly.csv'  
df = pd.read_csv(original_file)
Weekly_data = []

for idx, row in df.iterrows():
    for week in range(1, 53):
        Weekly_data.append({
            'Year': row['Year'],
            'Week': week,
            'Rainfall_mm': (row['Rainfall_mm']/52) * np.random.uniform(0.8, 1.2),
            'Soil_Moisture': row['Soil_Moisture'] * np.random.uniform(0.9, 1.1),
            'Temperature_C': row['Temperature_C'] + np.random.uniform(-2, 2),
            'Crop_Type': row['Crop_Type'],
            'Fertilizer_kg_per_ha': row['Fertilizer_kg_per_ha'],
            'Crop_Yield_tons_per_ha': row['Crop_Yield_tons_per_ha']
        })

df_weekly = pd.DataFrame(Weekly_data)
print(df_weekly.head())
df_weekly.to_csv('Kitui_Weekly.csv', index=False)


file_path = 'Kitui_Weekly.csv'
df = pd.read_csv(file_path)

print("\nDataset Preview:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\nStandardized Column Names:")
print(df.columns.tolist())


print("\nMissing Values per Column:")
print(df.isnull().sum())


for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            fill_value = df[col].mode()[0]
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled missing categorical values in '{col}' with mode: {fill_value}")
        else:
            fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled missing numeric values in '{col}' with median: {fill_value}")


duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate rows removed.")


for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
            print(f"Converted '{col}' to numeric.")
        except:
            pass  


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Outlier Detection - {col}')
    plt.show()


for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    print(f"Capped outliers in '{col}' using IQR method.")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\nNumeric features normalized.")

data = pd.read_csv('Kitui_Weekly.csv')
if data['Crop_Type'].dtype == 'object':
    le = LabelEncoder()
    data['Crop_Type'] = le.fit_transform(data['Crop_Type'])

features = ['Rainfall_mm', 'Soil_Moisture', 'Temperature_C', 'Fertilizer_kg_per_ha', 'Crop_Type']
target = 'Crop_Yield_tons_per_ha'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = pd.read_csv('Kitui_Weekly.csv')


le = LabelEncoder()
df['Crop_Type_Encoded'] = le.fit_transform(df['Crop_Type'])

X = df[['Year', 'Week', 'Rainfall_mm', 'Soil_Moisture', 'Temperature_C', 'Fertilizer_kg_per_ha', 'Crop_Type_Encoded']]
y = df['Crop_Yield_tons_per_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\nStandardized Column Names:")
print(df.columns.tolist())

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            fill_value = df[col].mode()[0]
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled missing categorical values in '{col}' with mode: {fill_value}")
        else:
            fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled missing numeric values in '{col}' with median: {fill_value}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Outlier Detection - {col}')
    plt.show()

numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if 'rainfall' in df.columns and 'crop_yield' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df['rainfall'], y=df['crop_yield'])
    plt.title('Rainfall vs Crop Yield')
    plt.xlabel('Rainfall')
    plt.ylabel('Crop Yield')
    plt.grid(True)
    plt.show()

if 'year' in df.columns and 'rainfall' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='year', y='rainfall')
    plt.title('Rainfall Trend over Years')
    plt.xlabel('Year')
    plt.ylabel('Rainfall')
    plt.grid(True)
    plt.show()
    
if 'year' in df.columns and 'crop_yield' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='year', y='crop_yield')
    plt.title('Crop Yield Trend over Years')
    plt.xlabel('Year')
    plt.ylabel('Crop Yield')
    plt.grid(True)
    plt.show()

if 'crop_type' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='crop_type', y='crop_yield_tons_per_ha', data=df)
    plt.title('Crop Yield Distribution per Crop Type')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['rainfall_mm'], y=df['crop_yield_tons_per_ha'])
plt.title('Rainfall vs Crop Yield')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Crop Yield (tons per ha)')
plt.grid(True)
plt.show()

plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.hist(rf_residuals, bins=30, color='blue', alpha=0.7)
plt.title('Random Forest Residuals')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1,3,2)
plt.hist(xgb_residuals, bins=30, color='green', alpha=0.7)
plt.title('XGBoost Residuals')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1,3,3)
plt.hist(rnn_residuals, bins=30, color='orange', alpha=0.7)
plt.title('RNN Residuals')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)

plt.suptitle('Residuals Distribution for Random Forest, XGBoost, and RNN', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].scatter(y_test, rf_preds, color='blue', alpha=0.6)
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 45-degree line
axs[0].set_title('Random Forest: Actual vs Predicted')
axs[0].set_xlabel('Actual Crop Yield (tons/ha)')
axs[0].set_ylabel('Predicted Crop Yield (tons/ha)')
axs[0].grid(True)

axs[1].scatter(y_test, xgb_preds, color='green', alpha=0.6)
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[1].set_title('XGBoost: Actual vs Predicted')
axs[1].set_xlabel('Actual Crop Yield (tons/ha)')
axs[1].set_ylabel('Predicted Crop Yield (tons/ha)')
axs[1].grid(True)

axs[2].scatter(y_test, rnn_preds, color='orange', alpha=0.6)
axs[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[2].set_title('RNN: Actual vs Predicted')
axs[2].set_xlabel('Actual Crop Yield (tons/ha)')
axs[2].set_ylabel('Predicted Crop Yield (tons/ha)')
axs[2].grid(True)

fig.suptitle('Actual vs Predicted Crop Yield (RF, XGB, RNN)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

fig.savefig('Actual_vs_Predicted_All_Models.png', dpi=300)

plt.show()

correlation_matrix = df[['rainfall_mm', 'soil_moisture', 'temperature_c', 'fertilizer_kg_per_ha', 'crop_yield_tons_per_ha']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Selected Numeric Features')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='neg_mean_absolute_error')
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
print("\nBest Random Forest Parameters:", grid_search_rf.best_params_)

best_rf_predictions = best_rf_model.predict(X_test)

best_rf_mae = mean_absolute_error(y_test, best_rf_predictions)
best_rf_rmse = np.sqrt(mean_squared_error(y_test, best_rf_predictions)) 
best_rf_r2 = r2_score(y_test, best_rf_predictions)

print("\nBest Random Forest Model Evaluation (after tuning):")
print(f"MAE: {best_rf_mae:.4f}")
print(f"RMSE: {best_rf_rmse:.4f}")
print(f"R²: {best_rf_r2:.4f}")

le = LabelEncoder()
df_weekly['Crop_Type_Encoded'] = le.fit_transform(df_weekly['Crop_Type'])

X_weekly = df_weekly[['Year', 'Week', 'Rainfall_mm', 'Soil_Moisture', 'Temperature_C', 'Fertilizer_kg_per_ha', 'Crop_Type_Encoded']]
y_weekly = df_weekly['Crop_Yield_tons_per_ha']

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_weekly, y_weekly, test_size=0.2, random_state=42)

rf_weekly = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_weekly.fit(X_train_w, y_train_w)
y_pred_rf_w = rf_weekly.predict(X_test_w)

print("\nRandom Forest Results:")
print(f"R2: {r2_score(y_test_w, y_pred_rf_w):.4f}")
print(f"MAE: {mean_absolute_error(y_test_w, y_pred_rf_w):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_w, y_pred_rf_w)):.4f}")

xgb_weekly = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42, objective='reg:squarederror')
xgb_weekly.fit(X_train_w, y_train_w)
y_pred_xgb_w = xgb_weekly.predict(X_test_w)

print("\nXGBoost Results:")
print(f"R2: {r2_score(y_test_w, y_pred_xgb_w):.4f}")
print(f"MAE: {mean_absolute_error(y_test_w, y_pred_xgb_w):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_w, y_pred_xgb_w)):.4f}")

predictions = list(rainfall_scaled[-seq_length:])
rnn_forecast = []
for _ in range(11):
    input_seq = np.array(predictions[-seq_length:]).reshape((1, seq_length, 1))
    next_pred = model_rnn.predict(input_seq, verbose=0)[0][0]
    predictions.append([next_pred])
    rnn_forecast.append(next_pred)

rnn_forecast_real = scaler.inverse_transform(np.array(rnn_forecast).reshape(-1, 1)).flatten()

rf_prediction = rf_weekly.predict(X_new)
print(f"Predicted Crop Yield (Random Forest): {rf_prediction[0]:.2f} tons/ha")

predictions = rf_weekly.predict(X_new_multiple)
print(predictions)

new_data_multiple = {
    'Year': [2025, 2025, 2025],
    'Week': [10, 11, 12],
    'Rainfall_mm': [15.5, 14.2, 16.0],
    'Soil_Moisture': [0.45, 0.48, 0.44],
    'Temperature_C': [27.0, 26.5, 28.0],
    'Fertilizer_kg_per_ha': [50, 50, 50],
    'Crop_Type': ['Maize', 'Sorghum', 'Pigeon Peas']
}

new_df_multiple = pd.DataFrame(new_data_multiple)
new_df_multiple['Crop_Type_Encoded'] = le.transform(new_df_multiple['Crop_Type'])

X_new_multiple = new_df_multiple[['Year', 'Week', 'Rainfall_mm', 'Soil_Moisture', 'Temperature_C', 'Fertilizer_kg_per_ha', 'Crop_Type_Encoded']]

xgb_yield_predictions = xgb_weekly.predict(X_new_multiple)

print("Predicted Crop Yields (XGBoost):")
print(xgb_yield_predictions)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': cb.CatBoostRegressor(iterations=100, random_state=42, verbose=0)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2
    })

best_rf = RandomForestRegressor(**study_rf.best_params, random_state=42)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nPerformance of Tuned Random Forest:")
print(f"MAE: {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R2 Score: {r2_rf:.4f}")

def xgb_objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    model = xgb.XGBRegressor(**param, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

print("\nStarting XGBoost Tuning...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgb_objective, n_trials=50)

print("Best Parameters for XGBoost:")
print(study_xgb.best_params)

best_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=42)
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nPerformance of Tuned XGBoost:")
print(f"MAE: {mae_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"R2 Score: {r2_xgb:.4f}")

rf_preds = best_rf.predict(X_test)
rf_residuals = y_test.values - rf_preds

xgb_preds = best_xgb.predict(X_test)
xgb_residuals = y_test.values - xgb_preds

rnn_preds_scaled = model_rnn.predict(X_test_rnn, verbose=0)
rnn_preds = scaler_y.inverse_transform(rnn_preds_scaled).flatten()

rnn_residuals = y_test.values - rnn_preds

X = df[['Year', 'Week', 'Rainfall_mm', 'Soil_Moisture', 'Temperature_C', 'Fertilizer_kg_per_ha', 'Crop_Type_Encoded']]
y = df['Crop_Yield_tons_per_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                         random_state=42, objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
rf_base_preds = rf_base.predict(X_test)
store_metrics(y_test, rf_base_preds, 'Random Forest (Before Tuning)')

rf_preds = rf_model.predict(X_test)
store_metrics(y_test, rf_preds, 'Random Forest (After Tuning)')
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.scatter(rf_preds, rf_residuals, alpha=0.6, color='blue')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Random Forest Residuals')
plt.xlabel('Predicted Yield (tons/ha)')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid(True)



xgb_base = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
xgb_base.fit(X_train, y_train)
xgb_base_preds = xgb_base.predict(X_test)
store_metrics(y_test, xgb_base_preds, 'XGBoost (Before Tuning)')

xgb_preds = xgb_model.predict(X_test)
store_metrics(y_test, xgb_preds, 'XGBoost (After Tuning)')

plt.subplot(1,2,2)
plt.scatter(xgb_preds, xgb_residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('XGBoost Residuals')
plt.xlabel('Predicted Yield (tons/ha)')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid(True)

plt.tight_layout()
plt.show()

model_rnn_base = Sequential()
model_rnn_base.add(SimpleRNN(10, activation='tanh', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
model_rnn_base.add(Dense(1))
model_rnn_base.compile(optimizer='adam', loss='mse')
model_rnn_base.fit(X_train_rnn, y_train_scaled, epochs=50, verbose=0)

rnn_base_preds_scaled = model_rnn_base.predict(X_test_rnn, verbose=0)
rnn_base_preds = scaler_y.inverse_transform(rnn_base_preds_scaled).flatten()

store_metrics(y_test, rnn_base_preds, 'RNN (Before Tuning)')

rnn_preds_scaled = model_rnn.predict(X_test_rnn, verbose=0)
rnn_preds = scaler_y.inverse_transform(rnn_preds_scaled).flatten()

store_metrics(y_test, rnn_preds, 'RNN (After Tuning)')
metrics_df = pd.DataFrame(results)

from sklearn.metrics import r2_score

def plot_actual_vs_predicted(y_true, y_pred, label="Model"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2) 
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    r2 = r2_score(y_true, y_pred)
    plt.title(f'Actual vs Predicted - {label}\n$R^2$ Score: {r2:.4f}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted(y_test, rf_preds, label="Random Forest")

plot_actual_vs_predicted(y_test, xgb_preds, label="XGBoost")

plot_actual_vs_predicted(y_test, rnn_preds, label="RNN")
plt.figure(figsize=(18,5))


fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(rf_residuals, kde=True, bins=30, color='blue', ax=axs[0], alpha=0.7)
axs[0].set_title('Random Forest Residuals')
axs[0].set_xlabel('Residual (Actual - Predicted)')
axs[0].set_ylabel('Frequency')
axs[0].grid(True)
sns.histplot(xgb_residuals, kde=True, bins=30, color='green', ax=axs[1], alpha=0.7)
axs[1].set_title('XGBoost Residuals')
axs[1].set_xlabel('Residual (Actual - Predicted)')
axs[1].set_ylabel('Frequency')
axs[1].grid(True)
sns.histplot(rnn_residuals, kde=True, bins=30, color='orange', ax=axs[2], alpha=0.7)
axs[2].set_title('RNN Residuals')
axs[2].set_xlabel('Residual (Actual - Predicted)')
axs[2].set_ylabel('Frequency')
axs[2].grid(True)
fig.suptitle('Residuals Distribution for RF, XGB, and RNN', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('Residuals_Distribution_All_Models.png', dpi=300)
plt.show()