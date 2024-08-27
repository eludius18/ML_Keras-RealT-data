from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load the dataset from a CSV file
    df = pd.read_csv('./data/kaggle_house_data.csv')

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract the month and year from the 'date' column
    df['month'] = df['date'].apply(lambda date: date.month)
    df['year'] = df['date'].apply(lambda date: date.year)

    # Drop the 'date' column as it's no longer needed
    df = df.drop('date', axis=1)

    # Separate the features (X) and the target variable (y)
    X = df.drop('price', axis=1)
    y = df['price']

    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(30, activation='relu', input_shape=input_shape))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mse')
    return model

def train_model():
    X, y = load_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Print the shapes of the training and testing data
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Build the model
    model = build_model((X_train.shape[1],))

    # Train the model
    model.fit(x=X_train, y=y_train.values, validation_data=(X_test, y_test.values), batch_size=128, epochs=400)
    print("Model trained")

    # Convert the training history to a DataFrame
    losses = pd.DataFrame(model.history.history)

    # Plot the training losses
    losses.plot()

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

    # Plot predictions vs actual values
    plt.scatter(y_test, predictions)
    plt.plot(y_test, y_test, 'r')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Predicted vs Actual Prices")
    plt.show()

    # Calculate and print the errors
    errors = y_test.values.reshape(-1, 1) - predictions

    # Plot the distribution of errors using histplot
    sns.histplot(errors, kde=True)
    plt.title("Distribution of Prediction Errors")
    plt.show()

    # Save the model and the scaler
    model.save('house_price_model.keras')
    return model, scaler

if __name__ == "__main__":
    train_model()
