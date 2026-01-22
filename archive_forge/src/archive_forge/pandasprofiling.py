import pandas as pd
from pandas_profiling import ProfileReport

# Load the dataset
data = pd.read_csv("/home/lloyd/Downloads/exampledata/example.csv")

# Generate data profile report
profile = ProfileReport(data, title="Data Profile Report")

# Save the report as an HTML file
profile.to_file("data_profile_report.html")
