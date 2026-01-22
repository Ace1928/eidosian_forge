import great_expectations as ge

# Load the dataset
data = ge.read_csv("/home/lloyd/Downloads/exampledata/example.csv")

# Define expectations
expectations = [
    {
        "expectation_type": "expect_column_values_to_not_be_null",
        "kwargs": {"column": "age"},
    },
    {
        "expectation_type": "expect_column_values_to_be_between",
        "kwargs": {"column": "age", "min_value": 18, "max_value": 100},
    },
    {
        "expectation_type": "expect_column_values_to_be_in_set",
        "kwargs": {"column": "gender", "value_set": ["Male", "Female"]},
    },
]

# Validate the dataset against expectations
validation_result = data.validate(expectations)

# Print validation results
print(validation_result)

# Generate a validation report
validation_result.save_as_html(
    "/home/lloyd/Downloads/exampledata/validation_report.html"
)
