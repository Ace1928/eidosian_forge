from aequitas.group import Group
from aequitas.bias import Bias

# Load the dataset and model predictions
data = ...  # Dataset with features, target, and sensitive attributes
predictions = ...  # Model predictions

# Create a group object
group = Group()
group.fit(data, sensitive_attributes=["race", "gender"])

# Create a bias object
bias = Bias()
bias.fit(group, predictions)

# Calculate fairness metrics
fairness_metrics = bias.get_disparity_major_group(
    group_metrics=["tpr", "fpr", "precision", "recall"]
)

# Print the fairness metrics
print(fairness_metrics)
