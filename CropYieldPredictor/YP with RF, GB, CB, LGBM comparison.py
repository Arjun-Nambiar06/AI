import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    categorical_cols = ["District", "Crop", "Season", "Seed_Quality", "Mechanization"]
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    return df

path = "Crop_Yield_Dataset.csv"

df = load_data(path)

X = df.drop("Yield_kg_ha", axis=1)
y = df["Yield_kg_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
cb_model = CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
cb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

cb_model.save_model("catboost_crop_yield.cbm")
joblib.dump(cb_model, "catboost_crop_yield.pkl")
print("CatBoost model saved as 'catboost_crop_yield.cbm' and 'catboost_crop_yield.pkl'")

y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_cb = cb_model.predict(X_test)
y_pred_lgbm = lgbm_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"=== {model_name} Results ===")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%\n")

evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_cb, "CatBoost")
evaluate_model(y_test, y_pred_lgbm, "LightGBM")

rf_importances = rf_model.feature_importances_
gb_importances = gb_model.feature_importances_
cb_importances = cb_model.feature_importances_
lgbm_importances = lgbm_model.feature_importances_
features = X.columns

fig, axes = plt.subplots(2, 2, figsize=(14,10))

axes[0,0].barh(features, rf_importances, color='skyblue')
axes[0,0].set_title("Random Forest Feature Importance")
axes[0,0].set_xlabel("Importance")

axes[0,1].barh(features, gb_importances, color='lightgreen')
axes[0,1].set_title("Gradient Boosting Feature Importance")
axes[0,1].set_xlabel("Importance")

axes[1,0].barh(features, cb_importances, color='orange')
axes[1,0].set_title("CatBoost Feature Importance")
axes[1,0].set_xlabel("Importance")

axes[1,1].barh(features, lgbm_importances, color='purple')
axes[1,1].set_title("LightGBM Feature Importance")
axes[1,1].set_xlabel("Importance")

plt.tight_layout()
plt.show()

