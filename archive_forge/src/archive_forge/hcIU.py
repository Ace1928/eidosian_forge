import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("sales_data.csv")

# Create a bar chart for sales by category
plt.figure(figsize=(8, 6))
plt.bar(data["Category"], data["Sales"])
plt.xlabel("Category")
plt.ylabel("Sales")
plt.title("Sales by Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sales_by_category.png")

# Create a pie chart for sales distribution
plt.figure(figsize=(8, 8))
plt.pie(data["Sales"], labels=data["Category"], autopct="%1.1f%%")
plt.title("Sales Distribution")
plt.tight_layout()
plt.savefig("sales_distribution.png")
