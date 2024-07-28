
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

##Data Preparation
data = pd.read_csv("Mobile-Price-Prediction-cleaned_data.csv", delimiter=",")
data.shape
data.head(10)
sns.pairplot(data, diag_kind='kde')
plt.show()

# Get unique values and their counts for each variable except Price
unique_values_counts = {column: data[column].value_counts().sort_index(ascending=False) for column in data.columns if column != 'Price'}

# Print the unique values and their counts for each variable
for column, values in unique_values_counts.items():
    print(f"Unique values and counts in {column}:")
    print(values)
    print()

# Get the number of missing data points for each column
missing_data_counts = data.isna().sum()

# Print the number of missing data points for each column
print("Number of missing data points for each column:")
print(missing_data_counts)

# Define the columns and values to delete
columns_to_filter = {
    'RAM': [0],
    'Mobile_Size': [44]
}

# Create a boolean mask for filtering rows
mask = pd.Series([True] * len(data))
for column, values in columns_to_filter.items():
    mask &= ~data[column].isin(values)

# Apply the mask to filter the data
filtered_data = data[mask]

# Save the filtered dataset to a new CSV file
filtered_data.to_csv("Mobile-Price-Prediction-cleaned_data_filtered.csv", index=False)

# Print the first few rows of the filtered dataset to confirm changes
print(filtered_data.head())
filtered_data.shape