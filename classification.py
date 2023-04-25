import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV

data = pd.read_csv('processed_log1.csv')
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


data['targetclass'] = data['targetmood'].apply(lambda x: classify_mood(x, target_mood_mean, target_mood_std))
print('--------CLASSIFICATION-----------')

X = data.drop(['id','period', 'targetclass', 'targetmood'], axis=1)
y = data['targetclass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print(report)

print('--------NUMERICAL PREDICTIONS-----------')


def preprocess_period(data):
    # Split the period column into start and end dates
    data[['start_date', 'end_date']] = data['period'].str.split(' - ', expand=True)

    # Convert start_date and end_date columns to datetime objects
    data['start_date'] = pd.to_datetime(data['start_date'])
    data['end_date'] = pd.to_datetime(data['end_date'])

    # Calculate the duration of the period in days
    data['duration'] = (data['end_date'] - data['start_date']).dt.days

    # Extract the day of the week (0 = Monday, 1 = Tuesday, etc.) from the start_date and end_date columns
    data['start_day_of_week'] = data['start_date'].dt.dayofweek
    data['end_day_of_week'] = data['end_date'].dt.dayofweek

    # Drop the period, start_date, and end_date columns
    data.drop(['period', 'start_date', 'end_date'], axis=1, inplace=True)

    return data


data = preprocess_period(data)


print(data.head())
X = data.drop(['id','targetclass', 'targetmood'], axis=1)
y = data['targetmood']
# print(X.head())
# Perform RFE for feature selection
model = LinearRegression()
cv = KFold(n_splits=5)  # You can change the number of folds
rfecv = RFECV(model, step=1, cv=cv, scoring='neg_mean_squared_error')
rfecv = rfecv.fit(X, y)
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {X.columns[rfecv.support_]}")

# Use only the selected features
#X_selected = X[X.columns[rfecv.support_]]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the linear regression model
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
