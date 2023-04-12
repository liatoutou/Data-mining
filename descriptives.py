import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('dataset_mood_smartphone.csv')

numerical_vars = [
    'mood', 'circumplex.arousal', 'circumplex.valence', 'activity',
    'screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
    'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
    'appCat.utilities', 'appCat.weather'
]

# Variables that need log transformation

# log_variables = [ 'screen', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 
# 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 
# 'appCat.unknown', 'appCat.utilities']

# Filter out rows with negative values for all numerical variables, except for 'circumplex.arousal' and 'circumplex.valence'
filtered_numerical_vars = [var for var in numerical_vars if var not in ['circumplex.arousal', 'circumplex.valence']]
df = df[~((df['variable'].isin(filtered_numerical_vars)) & (df['value'] < 0))]

# Function to find the amount of datapoints for each variable
def find_datapoints(df, variable_name):
    filtered_data = df[df['variable'] == variable_name]
    return len(filtered_data)

# Function to find the amount of missing values for each variable
def find_missing_values(df, variable_name):
    filtered_data = df[df['variable'] == variable_name]
    return filtered_data['value'].isnull().sum()

# Function to find the range of values for each variable
def find_range(df, variable_name):
    filtered_data = df[df['variable'] == variable_name]
    min_value = filtered_data['value'].min()
    max_value = filtered_data['value'].max()
    return min_value, max_value

# Function to find the average, median, and standard deviation of values for each variable
def find_stats(df, variable_name):
    filtered_data = df[df['variable'] == variable_name]
    mean = filtered_data['value'].mean()
    median = filtered_data['value'].median()
    std = filtered_data['value'].std()
    return mean, median, std


# Function to check the amount of outliers for each variable
def find_outliers(df, variable_name):
    # if variable_name in log_variables:
    #     # Transform log to make the outlier detection more accurate
    #     df['value'] = np.log1p(df['value'])
    filtered_data = df[df['variable'] == variable_name]
    q1 = filtered_data['value'].quantile(0.25)
    q3 = filtered_data['value'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = filtered_data[(filtered_data['value'] < lower_bound) | (filtered_data['value'] > upper_bound)]
    # df['value'] = np.expm1(df['value'])
    return outliers

# Iterate through variables and find the range for each
for variable_name in numerical_vars:
    print(f"{variable_name}:")
    datapoints = len(df[df['variable'] == variable_name])
    print(f"Datapoints = {datapoints}")
    empty_values = df[df['variable'] == variable_name]['value'].isnull().sum()
    print(f"Empty values = {empty_values}")
    mean, median, std = find_stats(df, variable_name)
    print(f"Mean = {mean}, Median = {median}, Standard Deviation = {std}")
    min_value, max_value = find_range(df, variable_name)
    print(f"Range = {min_value} to {max_value}\n")
    outliers = find_outliers(df, variable_name)
    print(f"Outliers = {len(outliers)}\n")