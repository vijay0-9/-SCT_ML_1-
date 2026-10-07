# TASK 01 - HOUSE PRICE PREDICTION
# Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("house_price.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# 2. Select Features and Target
X = df[["square_footage", "bedrooms", "bathrooms"]]
y = df["price"]


# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# 4. Create Linear Regression Model
model = LinearRegression()


# 5. Train Model
model.fit(X_train, y_train)


# 6. Make Predictions
y_pred = model.predict(X_test)


# 7. Model Coefficients
print("\nModel Coefficients:")
print("Square Footage:", model.coef_[0])
print("Bedrooms:", model.coef_[1])
print("Bathrooms:", model.coef_[2])
print("Intercept:", model.intercept_)


# 8. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# 9. Compare Actual vs Predicted
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted:")
print(result.head(10))


# 10. Visualization
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


# 11. Predict Price for a New House

new_house = pd.DataFrame({
    "square_footage": [2000],
    "bedrooms": [3],
    "bathrooms": [2]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print(predicted_price[0])
