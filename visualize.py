import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df = pd.read_csv("dataset_mood_smartphone.csv")
# Define the id and variables to plot
ids = [f"AS14.{i:02d}" for i in range(1, 34)]
#ids = ["AS14.15"]
'''variables = [
    'mood', 'circumplex.arousal', 'circumplex.valence', 'activity',
    'screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
    'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
    'appCat.utilities', 'appCat.weather'
]'''
variables = ['screen']
for id_ in ids:
    # Filter the DataFrame to keep only rows with the current id
    filtered_df = df[df['id'] == id_]

    # Create a pivot table with 'time' as index and 'variable' values as columns
    pivot_df = filtered_df.pivot_table(values='value', index='time', columns='variable', aggfunc='first')

    # Convert the index to a DatetimeIndex
    pivot_df.index = pd.to_datetime(pivot_df.index)

    # Resample the time series data to a daily frequency and compute the mean
    daily_resampled_df = pivot_df.resample('D').mean()

    # Iterate through the variables and create a plot for each
    for var in variables:
        # Check if the current variable exists for the specified id
        if var in daily_resampled_df.columns:
            # Plot the daily mean variable against time
            fig, ax = plt.subplots()
            ax.plot(daily_resampled_df.index, daily_resampled_df[var])

            # Set x-axis tick formatting and spacing
            date_format = mdates.DateFormatter('%Y-%m-%d')
            date_locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(date_locator)

            # Set axis labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel(var)
            ax.set_title(f'Daily mean {var} over time for id {id_}')

            # Rotate x-axis ticks for better readability
            plt.xticks(rotation=45)
            plt.show()

            # Save the plot as an image file
           # plt.savefig(f'{id_}_{var}_plot.png', bbox_inches='tight')

            # Close the plot to free up memory
            #plt.close(fig)