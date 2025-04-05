from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = pd.read_csv("W119preprocessed99.csv") # no lables

TARGET = 'AIHCCOMF_W119'
FEATURES = [
    'F_AGECAT', 'USEAI_W119',
    'AIWRK2_a_W119', 'AIWRK2_b_W119', 'AIWRK2_c_W119', 'AIWRK3_a_W119', 'AIWRK3_b_W119', 'AIWRK3_c_W119', 'AIWRKH1_W119', 'AIWRKH2_a_W119', 'AIWRKH2_b_W119',
    'AIWRKH3_a_W119', 'AIWRKH3_b_W119', 'AIWRKH3_c_W119', 'AIWRKH3_d_W119', 'AIWRKH4_W119', 'AIWRKM1_W119', 'AIWRKM2_a_W119', 'AIWRKM2_b_W119', 'AIWRKM2_c_W119',
    'AIWRKM2_d_W119', 'AIWRKM2_e_W119', 'AIWRKM2_f_W119', 'AIWRKM3_a_W119', 'AIWRKM3_b_W119', 'AIWRKM3_c_W119', 'AIWRKM3_d_W119', 'AIWRKM3_e_W119', 'AIWRKM3_f_W119',
    'AIWRKM4_a_W119', 'AIWRKM4_b_W119'
]

df = df[FEATURES + [TARGET]].copy()
df = df.loc[:, ~df.columns.str.contains('_OE|_REFUSED')]

NUMERICAL_FEATURES = [
    'USEAI_W119', 'AIWRK2_a_W119', 'AIWRK2_b_W119', 'AIWRK2_c_W119', 'AIWRK3_a_W119', 'AIWRK3_b_W119', 'AIWRK3_c_W119', 'AIWRKH1_W119', 'AIWRKH2_a_W119', 
    'AIWRKH2_b_W119', 'AIWRKH3_a_W119', 'AIWRKH3_b_W119', 'AIWRKH3_c_W119', 'AIWRKH3_d_W119', 'AIWRKM1_W119', 'AIWRKM2_a_W119', 'AIWRKM2_b_W119', 'AIWRKM2_c_W119', 
    'AIWRKM2_d_W119', 'AIWRKM2_e_W119', 'AIWRKM2_f_W119', 'AIWRKM3_a_W119', 'AIWRKM3_b_W119', 'AIWRKM3_c_W119', 'AIWRKM3_d_W119', 'AIWRKM3_e_W119', 'AIWRKM3_f_W119',
    'AIWRKM4_a_W119', 'AIWRKM4_b_W119']

CATEGORICAL_FEATURES = ['F_AGECAT', 'AIWRKH4_W119']

df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].fillna(df[NUMERICAL_FEATURES].median())
df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna('Unknown')

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), NUMERICAL_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)])

X = df.drop(TARGET, axis=1)
y = df[TARGET]
X_processed = preprocessor.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, 
    test_size=0.2, 
    random_state=42)

# train the model
model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=20,
    min_impurity_decrease=0.01,
    random_state=42)
model.fit(X_train, y_train)

# Save the model and preprocessor
joblib.dump(model, 'workplace_ai_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\nModel and preprocessor saved successfully!")

y_pred = model.predict(X_test)

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).round(3)  
results = results.sort_index()
print("Actual vs Predicted Values:")
print(results.head(10))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
all_features = NUMERICAL_FEATURES + list(cat_features)

feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
})
feature_importance['Importance'] = feature_importance['Importance'].round(3) 
feature_importance = feature_importance.sort_values('Importance', ascending=False)


print("\nTop 5 Important Features: ")
print(feature_importance.head(5))
