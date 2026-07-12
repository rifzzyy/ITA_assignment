import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("crop_yield.csv")

# Remove empty columns
data = data.drop(columns=['Unnamed: 13', 'Unnamed: 14'])

# Convert crop type into numerical values
data = pd.get_dummies(data, columns=['crop_type'])

# Display dataset information
print("\n🌱 Crop Yield Prediction using Decision Tree")
print("-------------------------------------------")
print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(list(data.columns))

print("\nMissing Values:")
print(data.isnull().sum())

# Select input features and target
X = data.drop(columns=['field_id', 'date_of_image', 'yield'])
y = data['yield']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# Create Decision Tree model
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")

# Predict crop yield
predictions = model.predict(X_test)

# Display predictions
print("\nFirst 10 Predicted Crop Yields")
print(predictions[:10])

print("\nFirst 10 Actual Crop Yields")
print(y_test.values[:10])

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n📊 Model Performance")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R² Score:", round(r2, 2))

# Sample prediction
sample = X_test.iloc[[0]]
sample_prediction = model.predict(sample)

print("\n🌾 Sample Prediction")
print("Predicted Yield :", round(sample_prediction[0], 2))
print("Actual Yield    :", round(y_test.iloc[0], 2))

print("\n🎉 Crop Yield Prediction Completed Successfully!")