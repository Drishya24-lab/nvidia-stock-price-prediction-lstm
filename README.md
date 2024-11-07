# Nvidia Stock Price Prediction Using LSTM and Bidirectional LSTM

This project uses historical stock price data for Nvidia to predict future closing prices using deep learning models, specifically LSTM (Long Short-Term Memory) and Bidirectional LSTM. The project involves data preprocessing, feature scaling, model building, and evaluation.

## Project Overview

In this project, I will predict Nvidia stock prices using historical data. I will use the following models:
- LSTM (Long Short-Term Memory)
- Bidirectional LSTM

## Table of Contents
1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Building](#model-building)
   - [LSTM Model](#lstm-model)
   - [Bidirectional LSTM Model](#bidirectional-lstm-model)
4. [Evaluation](#evaluation)
5. [Visualization](#visualization)

## Dataset

The dataset used is `nvidia_stock_prices.csv`, which contains daily stock prices for Nvidia. The key columns are:

- `Date`: The date of the stock price.
- `Open`, `High`, `Low`, `Close`, `Volume`: Stock price data for each day.

## Data Preprocessing

### 1. Loading the Data
The dataset is loaded using Pandas. The `Date` column is converted to datetime format and set as the index for easier time series analysis.

### 2. Feature Scaling
The `MinMaxScaler` is used to scale the stock prices between 0 and 1 for better performance of the LSTM model.

### 3. Train-Test Split
The dataset is split into training and testing data. 80% of the data is used for training, and the remaining 20% is used for testing.

### 4. Preparing the Data for LSTM
For LSTM models, the data is reshaped into sequences of 60 previous days (look-back window) to predict the next day's stock price.

## Model Building

### LSTM Model

A basic LSTM model is built with the following layers:
- LSTM layer with 50 units and a dropout layer to prevent overfitting.
- A second LSTM layer followed by a dropout layer.
- A Dense layer to output the prediction.

#### Model Compilation and Training

The model is compiled using the Adam optimizer and mean squared error as the loss function. The model is trained for 20 epochs.

### Bidirectional LSTM Model

For improved performance, a Bidirectional LSTM model is built:
- A bidirectional LSTM layer with 100 units.
- Another LSTM layer with 100 units.
- A dropout layer after each LSTM layer.

The model is trained with early stopping to prevent overfitting.

## Evaluation

### Performance Metrics

The performance of the model is evaluated using the Root Mean Squared Error (RMSE).

#### RMSE for Basic LSTM Model:
2.4950179266839316
### RMSE for Improved Bidirectional LSTM Model:
1.2309514573294795
### Conclusion

In this project, we successfully built two deep learning models, **LSTM** and **Bidirectional LSTM**, to predict Nvidia stock prices based on historical data. 

Here’s a summary of the results:

- **LSTM Model**: The basic LSTM model performed reasonably well with an RMSE of **2.495**. This indicates that the model was able to predict the stock prices with some level of accuracy, but there is room for improvement.
  
- **Bidirectional LSTM Model**: The improved **Bidirectional LSTM** model significantly outperformed the basic LSTM model, achieving a lower RMSE of **1.230**. The bidirectional approach allows the model to consider both past and future stock prices in its predictions, leading to better accuracy.

#### Key Takeaways:
- The performance improvement with the Bidirectional LSTM suggests that utilizing both past and future context can capture trends in stock prices more effectively.
- While the models demonstrated predictive power, stock price prediction is inherently volatile, and further improvements can be made by tuning hyperparameters, using more features, or incorporating additional data sources (e.g., news sentiment analysis or external economic indicators).

#### Future Work:
- Experiment with more advanced models such as **GRU**, **Attention Mechanisms**, or **Transformer-based models** to further improve prediction accuracy.
- Explore the use of external data such as market sentiment or news to enhance the model’s predictive power.




