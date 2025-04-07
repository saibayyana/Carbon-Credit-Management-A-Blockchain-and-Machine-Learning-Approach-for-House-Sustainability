import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Define range for the number of households
num_households = np.random.randint(490, 550)

# Function to generate synthetic data
def generate_synthetic_data(num_entities, entity_type):
    data = []
    for i in range(num_entities):
        if entity_type == "Household":
            electricity = np.random.randint(200, 401)
            cooking_fuel = np.random.randint(5, 31)
            water_heating = np.random.randint(30, 61)
            waste = np.random.randint(20, 41)
            transportation = np.random.randint(300, 601)
            no_of_cylinders = round(cooking_fuel / 14, 3)

        data.append([
            f"{entity_type}_{i+1}",
            entity_type,
            electricity,
            cooking_fuel,
            no_of_cylinders,
            water_heating,
            waste,
            transportation
        ])
    return data

# Generate data for households
households_data = generate_synthetic_data(num_households, "Household")

# Combine all data into a single DataFrame
columns = ["ID", "Type", "Electricity (kWh/month)", "Cooking Fuel (kg/month)", 
           "No of Cylinders", "Water Heating (kWh/month)", "Waste (kg/month)", 
           "Private Transportation (km/month)"]
df = pd.DataFrame(households_data, columns=columns)

# Emission factors
ELECTRICITY_EF = 0.82
COOKING_LPG_EF = 2.983
TRANSPORT_PETROL_EF = 2.31
WASTE_EF = 0.2
FUEL_EFFICIENCY = 15

# Calculate emissions
df["Electricity Emissions (kg CO₂/month)"] = df["Electricity (kWh/month)"] * ELECTRICITY_EF
df["Cooking Emissions (kg CO₂/month)"] = df["Cooking Fuel (kg/month)"] * COOKING_LPG_EF
df["Transportation Emissions (kg CO₂/month)"] = (df["Private Transportation (km/month)"] / FUEL_EFFICIENCY) * TRANSPORT_PETROL_EF
df["Waste Emissions (kg CO₂/month)"] = df["Waste (kg/month)"] * WASTE_EF

# Calculate total emissions
df["Total Emissions (kg CO₂/month)"] = df[[  
    "Electricity Emissions (kg CO₂/month)",
    "Cooking Emissions (kg CO₂/month)",
    "Transportation Emissions (kg CO₂/month)",
    "Waste Emissions (kg CO₂/month)"
]].sum(axis=1)

# Add noise to total emissions to make the data more realistic
noise = np.random.normal(0, 10, size=len(df))
df["Total Emissions (kg CO₂/month)"] += noise

# Simulate AQI data
num_months = 1
aqi_value = np.random.randint(270, 300)  # Same AQI for all households
df["AQI"] = aqi_value  # Adding AQI as a feature

# Save the dataset with emissions
emission_data_path = r"C:\Users\leela\Capstone\synthetic_emission_data_with_emissions.csv"
df.to_csv(emission_data_path, index=False)
print(f"Dataset with emissions saved to {emission_data_path}")

# Features and target (including AQI as a feature)
X = df[["Electricity (kWh/month)", "Cooking Fuel (kg/month)", 
        "Water Heating (kWh/month)", "Waste (kg/month)", 
        "Private Transportation (km/month)", "AQI"]]
y = df["Total Emissions (kg CO₂/month)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regression": GradientBoostingRegressor(random_state=42),
    "Support Vector Regression": SVR(),
    "K-Nearest Neighbors Regression": KNeighborsRegressor()
}

# Train models and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "R²": r2}
    print(f"{name}:")
    print(f"  Mean Absolute Error: {mae:.3f}")
    print(f"  R-squared: {r2:.3f}")
    print()

# Identify the best model
best_model_name = min(results, key=lambda x: results[x]["MAE"])
best_model = models[best_model_name]
print(f"The best model is {best_model_name} with MAE: {results[best_model_name]['MAE']:.3f} and R²: {results[best_model_name]['R²']:.3f}")

# Predict emissions for the entire dataset using the best model
df["Predicted Emissions (kg CO₂/month)"] = best_model.predict(X)

# Save the synthetic data and predicted emissions
ml_output_path = r"C:\Users\leela\Capstone\ml_output.csv"
ml_output_df = df[["ID", "Total Emissions (kg CO₂/month)", "Predicted Emissions (kg CO₂/month)", "AQI"]]
ml_output_df.to_csv(ml_output_path, index=False)
print(f"ML output (synthetic data + predicted emissions) saved to {ml_output_path}")

# Plot MAE comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results.keys()), y=[results[model]["MAE"] for model in results], palette="viridis")
plt.xlabel("Regression Models", fontsize=12)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
plt.title("Comparison of Regression Models by Mean Absolute Error (MAE)", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

# Plot R² comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results.keys()), y=[results[model]["R²"] for model in results], palette="magma")
plt.xlabel("Regression Models", fontsize=12)
plt.ylabel("R-squared (R²)", fontsize=12)
plt.title("Comparison of Regression Models by R-squared (R²)", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()
