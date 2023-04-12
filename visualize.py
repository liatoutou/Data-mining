import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('dataset_mood_smartphone.csv')

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

numerical_vars = [
    'mood', 'circumplex.arousal', 'circumplex.valence', 'activity',
    'screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
    'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
    'appCat.utilities', 'appCat.weather'
]

# Filter out rows with negative values for all numerical variables, except for 'circumplex.arousal' and 'circumplex.valence'
filtered_numerical_vars = [var for var in numerical_vars if var not in ['circumplex.arousal', 'circumplex.valence']]
df = df[~((df['variable'].isin(filtered_numerical_vars)) & (df['value'] < 0))]


# Transform log to make the visualization more readable
df.loc[df['variable'] == 'screen', 'value'] = np.log1p(df.loc[df['variable'] == 'screen', 'value'])
df.loc[df['variable'] == 'appCat.builtin', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.builtin', 'value'])
df.loc[df['variable'] == 'appCat.communication', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.communication', 'value'])
df.loc[df['variable'] == 'appCat.entertainment', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.entertainment', 'value'])
df.loc[df['variable'] == 'appCat.finance', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.finance', 'value'])
df.loc[df['variable'] == 'appCat.game', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.game', 'value'])
df.loc[df['variable'] == 'appCat.office', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.office', 'value'])
df.loc[df['variable'] == 'appCat.other', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.other', 'value'])
df.loc[df['variable'] == 'appCat.social', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.social', 'value'])
df.loc[df['variable'] == 'appCat.travel', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.travel', 'value'])
df.loc[df['variable'] == 'appCat.unknown', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.unknown', 'value'])
df.loc[df['variable'] == 'appCat.utilities', 'value'] = np.log1p(df.loc[df['variable'] == 'appCat.utilities', 'value'])


df = df[~((df['variable'].isin(numerical_vars)) & (df['value'] < 0))]

for var in numerical_vars:
    plt.figure()
    sns.histplot(data=df[df['variable'] == var], x='value')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()


# 'call', 'sms',

# plt.figure()
# sns.boxplot(data=app_cat_df, x='variable', y='value')
# plt.title('Boxplot of App Categories')
# plt.xlabel('App Category')
# plt.ylabel('Duration')
# plt.xticks(rotation=90)
# plt.show()

# # Pivot the dataset to wide format
# df_wide = df.pivot_table(index=['id', 'time'], columns='variable', values='value').reset_index()

# # Drop the 'id' and 'time' columns as they are not needed for the correlation analysis
# df_wide = df_wide.drop(columns=['id', 'time'])

# # Compute the correlation matrix
# correlation_matrix = df_wide.corr()

# # Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("Correlation Matrix")
# plt.show()
