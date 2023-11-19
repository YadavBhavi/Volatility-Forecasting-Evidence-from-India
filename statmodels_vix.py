import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
!pip install pmdarima
import pmdarima as pm
from IPython.display import clear_output
clear_output()
import pickle
import statsmodels.api as sm


df = pd.read_csv('vix_data.csv')

model = pm.auto_arima(df['NIFTY 30d volatility'], 
                        m=12, seasonal=True,
                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  
                           suppress_warnings=True, stepwise=True, trace=True)

pickle.dump(model, open('arima.sav', 'wb'))

train = df.iloc[:800]
test = df.iloc[800:]

model.fit(train['NIFTY 30d volatility'])

##################################################

data = df

batch_size = 240

# Define the ARIMA model parameters
p = 0
d = 0
q = 4
P = 2
D = 0
Q = 2
S = 12

# Initialize the model
model = None

# Initialize the MSE loss
mse_loss = 0

# Iterate over the data in batches
for i in range(0, len(data)-batch_size, batch_size):
    
    # Split the data into the current batch and the remaining data
    batch_data = data.iloc[i:i+batch_size]
    remaining_data = data.iloc[i+batch_size:]
    
    # Split the batch data into training and testing sets
    train_data = batch_data.iloc[:-1]
    test_data = batch_data.iloc[-1:]
    
    # Split the training data into endogenous and exogenous variables
    train_endog = train_data['NIFTY 30d volatility']
    train_exog = train_data.drop('NIFTY 30d volatility', axis=1)
    
    # Create the ARIMA model object if it hasn't been initialized yet
    if model is None:
        model = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=(p, d, q), seasonal_order=(P, D, Q, S))
    else:
        # Update the existing model with the new batch of data
        model.update(train_endog, train_exog)
    
    results = model.fit()
    # Generate predictions for the test data
    test_endog = test_data['NIFTY 30d volatility']
    test_exog = test_data.drop('NIFTY 30d volatility', axis=1)
    forecast = results.forecast(steps=len(test_data), exog=test_exog)
    
    # Calculate the MSE loss for the current batch and update the total MSE loss
    mse_loss += np.mean((test_endog - forecast)**2)

# Print the total MSE loss
print("Total MSE loss:", mse_loss)
