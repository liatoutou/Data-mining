import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('aggregated.csv')

specific_variables = ['mood', 'circumplex.valence', 'circumplex.arousal']
df_specific = df[df['variable'].isin(specific_variables)]

# Convert the 'date' column in df_specific to datetime64[ns]
df_specific['date'] = pd.to_datetime(df_specific['date'])

# Get the unique IDs
unique_ids = df['id'].unique()

# Initialize an empty DataFrame to store the updated data
updated_df = pd.DataFrame()

# Iterate through unique IDs
for uid in unique_ids:
    df_id = df_specific[df_specific['id'] == uid]

    # Find the first date and last date for each variable and ID
    first_date = df_id['date'].min()
    last_date = df_id.groupby(['id', 'variable'])['date'].max().max()

    # Create a date range from the first date to the last date
    date_range = pd.date_range(start=first_date, end=last_date)

    # Create a new DataFrame with a complete date range for each ID and variable
    complete_dates_df = pd.DataFrame()
    for var in specific_variables:
        temp_df = pd.DataFrame({'date': date_range, 'variable': var, 'id': uid})
        complete_dates_df = pd.concat([complete_dates_df, temp_df], ignore_index=True)

    # Merge the original DataFrame with the new DataFrame
    merged_df = pd.merge(complete_dates_df, df_id, on=['id', 'date', 'variable'], how='left')

    # Use the interpolate() function to fill the missing values
    merged_df['value'] = merged_df['value'].interpolate()

    # Add the processed DataFrame to the updated_df
    updated_df = pd.concat([updated_df, merged_df], ignore_index=True)


# Set the index of both DataFrames to the columns you want to match
df.set_index(['id', 'date', 'variable'], inplace=True)
updated_df.set_index(['id', 'date', 'variable'], inplace=True)

# Update the 'value' column in the original DataFrame with the new values, if the rows do not exist in df
df['value'] = df['value'].combine_first(updated_df['value'])

# Reset the index of the DataFrame
df.reset_index(inplace=True)

col_name = "Unnamed: 0"
first_col = df.pop(col_name)
df.insert(0, col_name, first_col)


# Save the updated DataFrame to a CSV file
df.to_csv('filled.csv', index=False)