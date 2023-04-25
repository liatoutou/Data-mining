import pandas as pd
from pmdarima import auto_arima
import re
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import numpy as np

# Load the dataset
data = pd.read_csv("arima.csv")

target_mood_values = data['targetmood']
target_mood_mean = target_mood_values.mean()
target_mood_std = target_mood_values.std()


def classify_mood(mood_value, mean, std):
    if mood_value < mean - std:
        return 'sad'
    elif mood_value > mean + std:
        return 'happy'
    else:
        return 'normal'


data['targetclass1'] = data['mood'].apply(lambda x: classify_mood(x, target_mood_mean, target_mood_std))
all_y = []
all_predictions = []
for unique_id in data['id'].unique():
    id_data = data[data['id'] == unique_id]  # Predict for each patient the mood of the next days
    targetmood_series = id_data['mood']
    categorical = id_data['targetclass1']
    # print(f"Target mood for {unique_id} is:", targetmood_series)
    # Split the time series into training and testing sets
    train_size = int(len(targetmood_series) * 0.8)  # Split the patient's data into train and test
    train, test = targetmood_series[:train_size], targetmood_series[train_size:]
    y_test = categorical[train_size:]
    for pred_str in y_test:
        pred_values = re.sub(r'\s*\d+\s*', '', pred_str)
        pred_list = pred_values.split(", ")
        all_y.extend(pred_list)

    # print(f"Train: {train}, Test: {test}")

    # Use auto_arima to find the best ARIMA parameters
    best_model = auto_arima(train, seasonal=False, stepwise=True,
                            suppress_warnings=True, trace=True,
                            error_action='ignore', information_criterion='aic')

    # print(f"Best model: {best_model}")

    # Fit the best ARIMA model
    best_model.fit(train)

    # Predict 'targetmood' for the test set
    predictions, conf_int = best_model.predict(n_periods=len(test), return_conf_int=True)
    # Assuming the predictions are in a list called 'predictions_list'
    for pred_series in predictions:
        if isinstance(pred_series, pd.Series):
            pred_values = pred_series.values.tolist()
        else:
            pred_values = [pred_series]
        all_predictions.extend(pred_values)

print(len(all_predictions))
print(len(all_y))

pred_mean = np.mean(all_predictions)

pred_std = np.std(all_predictions)
print(pred_mean,pred_std)
cat_preds = []
n=[]
for x in all_predictions:
    n.append(x)
    value = classify_mood(x,pred_mean,pred_std)
    print(value)
    cat_preds.append(value)

# Evaluate the model using Mean Squared Error

accuracy = accuracy_score(all_y, cat_preds)
report = classification_report(all_y, cat_preds)
print(f"Accuracy: {accuracy:.2f}")
print(report)
