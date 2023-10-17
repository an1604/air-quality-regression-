# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:20:31 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def calculate_good_air_quality_probability(co, nox, no2, benzene, ozone):
    # Define thresholds for each feature to indicate good air quality
    thresholds = {
        'CO(GT)': 1.0,  
        'NOx(GT)': 10,  
        'NO2(GT)': 20,  
        'C6H6(GT)': 2.0, 
        'PT08.S5(O3)': 1000  
    }

    # Calculate the probability of good air quality for each row
    # Higher values indicate a higher probability of good air quality
    good_air_quality_probabilities = []

    for i in range(len(co)):
        probability = 1.0  # Initialize probability for each row
        for feature, threshold in thresholds.items():
            if feature in ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)', 'PT08.S5(O3)']:
                if feature != 'PT08.S5(O3)':
                    probability *= min(1.0, co[i] / threshold)  # Update probability based on the ratio of the feature value to its threshold
                else:
                    probability *= min(1.0, ozone[i] / threshold)
        
        result = 1.0 - probability
        if result >=0:
            good_air_quality_probabilities.append(1.0 - probability)  # Probability of good air quality is 1 - calculated probability
        else:
            good_air_quality_probabilities.append(None)
    return good_air_quality_probabilities

# Importing the dataset 
dataset = pd.read_excel('AirQualityUCI.xlsx')

# Dropping the date and time to be more efficient 
new_ds = dataset[['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)', 'PT08.S5(O3)']]
new_ds  = pd.DataFrame(new_ds)
# getting the AQI rate 
air_quality = calculate_good_air_quality_probability(new_ds['CO(GT)'], new_ds['NOx(GT)'], new_ds['NO2(GT)'], new_ds['C6H6(GT)'], new_ds['PT08.S5(O3)'])

# dropping the nan values 
air_quality= [value for value in air_quality if value is not None]

# Calculate the number of rows for training and testing
train_rows = 7485
test_rows = 1872

# Split the dataset
dataset_train = new_ds[:train_rows]
dataset_test = new_ds[train_rows:]

# Reshape air quality arrays for scaling
air_quality_train = np.array(air_quality[:train_rows]).reshape(-1, 1)
air_quality_test = np.array(air_quality[train_rows:]).reshape(-1, 1)

# Feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

# Scaling the datasets (X_train, X_test)
training_set = sc.fit_transform(dataset_train)
test_set = sc.transform(dataset_test)


# Creating a data structure with 5 rows and 60 timesteps
X_train = []
y_train = []
for i in range(60, 7485):
    X_train.append([training_set[i-60:i, j] for j in range(5)])
    y_train.append(training_set[i, :13])  

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape X_train to have 5 rows and 60 columns
X_train = X_train.transpose(0, 2, 1)

# Building the RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],5)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 35, batch_size = 32)

# predictions 
real_AQI = air_quality_test[0:]

# making our X_test according the timesteps 
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,5)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,len(inputs)):
    X_test.append(inputs[i-60:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))


predictions = regressor.predict(X_test)


# Visualising the results - just for the first 250.
plt.plot(real_AQI, color = 'red', label = 'Real Air Quality Rate')
plt.plot(predictions[:250], color = 'blue', label = 'Predicted Air Quality Rate')
plt.title('Air Quality Rate Prediction')
plt.xlabel('Time')
plt.ylabel('Air Quality Rate')
plt.legend()
plt.show()
