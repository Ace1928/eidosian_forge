import pandas as pd
import os

# Define the directory containing the datasets
dataset_directory = "datasets"

# Create a data catalog DataFrame
data_catalog = pd.DataFrame(columns=["Dataset", "Rows", "Columns", "Size"])

# Iterate over the datasets in the directory
for dataset_file in os.listdir(dataset_directory):
    if dataset_file.endswith(".csv"):
        dataset_path = os.path.join(dataset_directory, dataset_file)
        dataset = pd.read_csv(dataset_path)

        dataset_name = os.path.splitext(dataset_file)[0]
        num_rows = len(dataset)
        num_columns = len(dataset.columns)
        dataset_size = os.path.getsize(dataset_path)

        data_catalog = data_catalog.append(
            {
                "Dataset": dataset_name,
                "Rows": num_rows,
                "Columns": num_columns,
                "Size": dataset_size,
            },
            ignore_index=True,
        )

# Save the data catalog as a CSV file
data_catalog.to_csv("data_catalog.csv", index=False)

print("Data catalog generated.")
