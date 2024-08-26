import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

# Load the dataset from a CSV file
df = pd.read_csv('./data/kc_house_data.csv')

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract the month and year from the 'date' column
df['month'] = df['date'].apply(lambda date: date.month)
df['year'] = df['date'].apply(lambda date: date.year)

# Drop the 'date' column from the DataFrame
df = df.drop('date', axis=1)

# Separate the features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the shapes of the training and testing data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Initialize the Sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(21, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(1))
print("Model architecture defined")

# Compile the model
model.compile(optimizer='adam', loss='mse')
print("Model compiled")

# Train the model
model.fit(x=X_train, y=y_train.values,
          validation_data=(X_test, y_test.values),
          batch_size=128, epochs=400)
print("Model trained")

# Convert the training history to a DataFrame
losses = pd.DataFrame(model.history.history)

# Plot the training losses
losses_plot = losses.plot()

# Print the test data
print("X_test:", X_test)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate and print the mean absolute error
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Calculate and print the root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error:", rmse)

# Calculate and print the explained variance score
evs = explained_variance_score(y_test, predictions)
print("Explained Variance Score:", evs)

# Print the mean price
mean_price = df['price'].mean()
print("Mean price:", mean_price)

# Print the median price
median_price = df['price'].median()
print("Median price:", median_price)

# Plot our predictions vs actual values
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')

# Calculate and print the errors
errors = y_test.values.reshape(-1, 1) - predictions

# Plot the distribution of errors using histplot
errors_plot = sns.histplot(errors, kde=True)
print(errors_plot)

# Select a single house from the DataFrame
single_house = df.drop('price', axis=1).iloc[0]

# Scale the single house data
single_house = scaler.transform(single_house.values.reshape(1, -1))

# Print the single house data
print("Single house data:", single_house)

# Make a prediction for the single house
single_house_prediction = model.predict(single_house)
print("Prediction for single house:", single_house_prediction)

# Print the details of the single house
single_house_details = df.iloc[0]
print("Single house details:", single_house_details)