import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df = pd.read_csv('used_cars.csv')
df = df.drop(columns=['name', 'torque'])
print(df.head())

# Preprocessing ===============================================================
# part 1 ----------------------------------------
cols_to_clean = ['mileage', 'engine', 'max_power']
for col in cols_to_clean:
    df[col] = df[col].str.split().str[0]
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(df.head())
print(df.info())

# part 2 ----------------------------------------
print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# part 3 ----------------------------------------
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}\n")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print(df_encoded.head())



# Visualization ===============================================================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
numeric_cols = ['selling_price', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='max_power', y='selling_price', alpha=0.5)
plt.title('Max Power vs Selling Price')

plt.tight_layout()
plt.show()



# Partitioning ================================================================
y = df_encoded['selling_price']
X = df_encoded.drop('selling_price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Regression Models ===========================================================
# Simple Linear Regression ----------------------
X_train_simple = X_train[['max_power']]
X_test_simple = X_test[['max_power']]

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)

y_pred_simple = simple_model.predict(X_test_simple)

mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print("Coef : ", simple_model.coef_[0])
print("Intercept : ", simple_model.intercept_)
print("R2-Score : ", r2_simple)

# Multiple Linear Regression --------------------
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)
mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)
print(f"R2-score: {r2_multi}")

# Polynomial Regression -------------------------
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
poly = PolynomialFeatures(degree=2, include_bias=False)

X_num_poly = poly.fit_transform(df[num_cols])
X_poly_df = pd.DataFrame(X_num_poly, columns=poly.get_feature_names_out(num_cols))

df_categorical = pd.get_dummies(df[['fuel', 'seller_type', 'transmission', 'owner']], drop_first=True, dtype=int)
X_full_poly = pd.concat([X_poly_df, df_categorical.reset_index(drop=True)], axis=1)

X_train_p, X_test_p, y_train, y_test = train_test_split(X_full_poly, y, test_size=0.2, random_state=42)

scaler_poly = StandardScaler()
X_train_p_scaled = scaler_poly.fit_transform(X_train_p)
X_test_p_scaled = scaler_poly.transform(X_test_p)

poly_model = LinearRegression()
poly_model.fit(X_train_p_scaled, y_train)
y_pred_poly = poly_model.predict(X_test_p_scaled)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print("R2-score:", r2_poly)

# Lasso, Ridge, ElasticNet ----------------------
# l1-Ratio = 0.5 Default
param_grid = {'alpha': [0.1, 1, 10, 100, 500]}

ridge = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
ridge.fit(X_train_p_scaled, y_train)

lasso = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=5, scoring='r2')
lasso.fit(X_train_p_scaled, y_train)

elastic = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5, scoring='r2')
elastic.fit(X_train_p_scaled, y_train)

print("--- BEST PARAMETERS ---")
print("Best params for Ridge:", ridge.best_params_['alpha'])
print("Best params for Lasso:", lasso.best_params_['alpha'])
print("Best params for ElasticNet:", elastic.best_params_['alpha'])



# Model Performance Evaluation & Comparison of the obtained results
y_pred_ridge = ridge.predict(X_test_p_scaled)
y_pred_lasso = lasso.predict(X_test_p_scaled)
y_pred_elastic = elastic.predict(X_test_p_scaled)

models = ['Simple Linear', 'Multiple Linear', 'Polynomial', 'Ridge', 'Lasso', 'ElasticNet']

r2_scores = [
    r2_simple, 
    r2_multi, 
    r2_poly, 
    r2_score(y_test, y_pred_ridge), 
    r2_score(y_test, y_pred_lasso), 
    r2_score(y_test, y_pred_elastic)
]

mse_scores = [
    mse_simple, 
    mse_multi, 
    mse_poly, 
    mean_squared_error(y_test, y_pred_ridge), 
    mean_squared_error(y_test, y_pred_lasso), 
    mean_squared_error(y_test, y_pred_elastic)
]

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.bar(models, r2_scores, color='teal')
plt.title('Comparison of Model Accuracy (R-squared)')
plt.ylabel('R-squared (Higher is Better)')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(models, mse_scores, color='crimson')
plt.title('Comparison of Model Error (MSE)')
plt.ylabel('Mean Squared Error (Lower is Better)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
