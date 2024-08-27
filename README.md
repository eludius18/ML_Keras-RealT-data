# House Price Prediction Model

## Overview
This project demonstrates a machine learning approach to predicting house prices using a dataset from Kaggle. The model is built using a neural network implemented with TensorFlow's Keras API. The dataset includes various features such as the date of sale, the size of the house, the number of bedrooms, and other attributes. The model aims to predict the house price based on these features.

## Technologies
- **Python**: The programming language used for this project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting and data visualization.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For data preprocessing and evaluation metrics.
- **TensorFlow / Keras**: For building and training the neural network model.

## How to Install

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/house-price-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd house-price-prediction
    ```
3. Install the required Python packages using pip:
    ```bash
    pip3 install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```

## How to Run

1. Ensure you have the Kaggle house data CSV file placed in the `./data/` directory with the filename `kaggle_house_data.csv`.

2. Run the script:
    ```bash
    python3 main.py
    ```

3. The script will:
    - Load and preprocess the data.
    - Split the data into training and testing sets.
    - Train a neural network model on the training data.
    - Evaluate the model's performance on the test data.
    - Generate various plots to visualize the model's performance and errors.
    - Make predictions on the test data and a single house's features.

4. The output will include:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Explained Variance Score (EVS)
    - Plots showing the loss during training, prediction vs. actual values, and error distribution.

## Dependencies

This project requires the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`

You can install these dependencies by running:
```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn tensorflow
```