# Air Quality Regression using RNN
## Air Quality Regression

This project utilizes Recurrent Neural Networks (RNN) to predict air quality levels based on several environmental factors. The dataset used in this project is 'AirQualityUCI.xlsx'.

# Prerequisites
To run this project, you will need the following dependencies:

Python (>= 3.6)
Jupyter Notebook or any Python IDE
Libraries: pandas, numpy, matplotlib, scikit-learn, Keras (with TensorFlow backend)
Project Overview
In this project, we aim to predict air quality levels based on the concentration of various air pollutants, such as carbon monoxide (CO), nitrogen oxide (NOx), nitrogen dioxide (NO2), benzene (C6H6), and ozone (O3). The model is designed to predict the probability of good air quality based on these pollutant concentrations.

# Dataset
The dataset 'AirQualityUCI.xlsx' is used for this project. It contains historical data for the selected air quality parameters.

# Data Preprocessing
The project begins by loading the dataset and selecting the relevant features: CO(GT), NOx(GT), NO2(GT), C6H6(GT), and PT08.S5(O3).
A custom function, calculate_good_air_quality_probability, is used to calculate the probability of good air quality based on predefined thresholds for each feature.
Rows with missing values are dropped.
Data Splitting
The dataset is split into training and testing sets. 80% of the data is used for training (7485 rows), and the remaining 20% is used for testing (1872 rows).

# Feature Scaling
Feature scaling is applied to both the training and testing datasets using Min-Max scaling.

# Model Architecture
The RNN model architecture consists of multiple LSTM layers followed by a Dense output layer.

Four LSTM layers with dropout regularization are used.
The output layer has a single unit to predict the air quality level.
The model is compiled with the Adam optimizer and mean squared error loss function.
# Model Training
The RNN model is trained on the training dataset with 35 epochs and a batch size of 32.

# Predictions
The trained model is used to make predictions on the test dataset. The predictions are compared to the real air quality values, and the results are visualized using a line chart.

# Visualization
The results are visualized in a line chart showing the real air quality rate in red and the predicted air quality rate in blue.
