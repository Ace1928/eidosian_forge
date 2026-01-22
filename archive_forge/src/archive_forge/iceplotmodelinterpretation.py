from pdpbox import pdp
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
X_train = ...  # Training features
y_train = ...  # Training labels

# Train a random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Select the feature to visualize
feature = "feature1"

# Create the ICE plots
ice = pdp.pdp_isolate(
    model=rf_regressor,
    dataset=X_train,
    model_features=X_train.columns,
    feature=feature,
    ice=True,
)

# Visualize the ICE plots
fig, ax = pdp.pdp_plot(ice, feature, plot_lines=True, figsize=(10, 5))
