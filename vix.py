import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

# Load the 5-year time series data of 8 variables into a pandas DataFrame
df = pd.read_csv('vix_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.fillna(method='bfill')
df = df.drop(['Date', '(R1) Open', '(R1) High', '(R1) Low'], axis=1)
df = df[::-1]
df.index = df.index[::-1]

# delete last row
df.drop(index=df.index[0], axis=0, inplace=True)


cols = [x for x in df.columns if x != 'NIFTY 30d volatility']
df[cols] = minmax_scale(df[cols])
df.drop(index=df.index[0], axis=0, inplace=True)
data = df.values
# delete last row
data = np.delete(data, -1, axis=0)


np.savetxt('vix_data.csv', data, delimiter=',', fmt='%.2f')

timesteps = 400  # number of timesteps in the input sequence (1 year)
features = 8  # number of features in each timestep
output_timesteps = 7  # number of timesteps in the output sequence (7 days)
n_shift = 50


def create_input_output_pairs(mydata, timesteps, features, output_timesteps, n_shift):
    X, y = [], []
    for i in range(len(mydata) - timesteps - output_timesteps + 1):
        if i % n_shift == 0:
            X.append(mydata[i:i+timesteps, :features])
            y.append(mydata[i+timesteps:i+timesteps+output_timesteps, 2])
    X = np.array(X)
    Y = np.array(y)
    print(X.shape, Y.shape)
    return X, Y


def make_train_test(X, y, train_percent=0.6):
    train_size = int(len(X) * train_percent)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    print(n_timesteps, n_features, n_outputs)
    return X_train, X_test, y_train, y_test


def model(n_timesteps, n_features, n_outputs):
    # Define the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(n_timesteps, n_features)))
    # model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(n_outputs))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Evaluate the model on the test data
def evaluate_model(mymodel, X_test, y_test):
    test_loss = mymodel.evaluate(X_test, y_test)
    print(f'Test loss: {test_loss}')
    # Use the trained model to make predictions
    predictions = mymodel.predict(X_test)
    print(predictions.shape)
    # plot_predictions(predictions, y_test)
    print("mean error when taking all features into account = ",(np.abs(predictions-y_test)).mean())
    return test_loss

X,y = create_input_output_pairs(data, 400, 8, 7, 50)
X_train, X_test, y_train, y_test = make_train_test(X, y, train_percent = 0.6)
model = model(X_train.shape[1], X_train.shape[2], y_train.shape[1])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2, callbacks=[callback])
# Save the model
model.save('vix_model.h5')
# Load the model
model_full = tf.keras.models.load_model('vix_model.h5')
main_loss = evaluate_model(model_full, X_test, y_test)

def ciop(data2, mydata, timesteps, features, output_timesteps, n_shift):
    X, y = [], []
    for i in range(len(mydata) - timesteps - output_timesteps + 1):
        if i % n_shift == 0:
            X.append(mydata[i:i+timesteps, :features])
            y.append(data2[i+timesteps:i+timesteps+output_timesteps])
    X = np.array(X)
    Y = np.array(y)
    print(X.shape, Y.shape)
    return X, Y

# now we will eliminate one feature at a time, revaluate input-output pairs, retrain the new model and plot the test loss vs feature eliminated

def loss_vs_eliminated_feature():
    yy = []
    for i in range(8):
        df2 = df.values
        data2 = df2[:,2]
        df1 = df.drop([df.columns[i]], axis=1)
        data1 = df1.values
        X1, y1 = ciop(data2, data1, 400, 7, 7, 50)
        # print(y1)
        X_train1, X_test1, y_train1, y_test1 = make_train_test(X1, y1, train_percent = 0.6)
        model1 = model(X_train1.shape[1], X_train1.shape[2], y_train1.shape[1])
        model1.fit(X_train1, y_train1, epochs=100, batch_size=1, verbose=0, callbacks=[callback])
        test_loss = evaluate_model(model1, X_test1, y_test1) - main_loss
        yy.append(test_loss)
    return yy

yy = loss_vs_eliminated_feature()
# plot the test loss vs feature eliminated
plt.plot(df.columns,yy)


# now we will eliminate two features at a time, revaluate input-output pairs, retrain the new model and plot the test loss vs features eliminated
df2 = df.values
data2 = df2[:,2]
df1 = df.drop([df.columns[1],df.columns[6]], axis=1)
data1 = df1.values
X1, y1 = ciop(data2, data1, 400, 6, 7, 50)
# print(y1)
X_train1, X_test1, y_train1, y_test1 = make_train_test(X1, y1,train_percent = 0.6)
model1 = model(X_train1.shape[1], X_train1.shape[2], y_train1.shape[1])
model1.fit(X_train1, y_train1, epochs=100, batch_size=1, verbose=0, callbacks=[callback])
evaluate_model(model1, X_test1, y_test1)

# plot test-loss vs no. of epoch (10-200) by training various models with different number of epochs

def plot_loss_vs_epochs(X_train, X_test, y_train, y_test, epochs):
    test_loss = []
    for i in epochs:
        model = model(X_train.shape[1], X_train.shape[2], y_train.shape[1])
        model.fit(X_train, y_train, epochs=i, batch_size=1, verbose=0, callbacks=[callback])
        test_loss.append(evaluate_model(model, X_test, y_test))
    plt.plot(epochs, test_loss)
    plt.xlabel('No. of epochs')
    plt.ylabel('Test loss')
    plt.show()
    return test_loss

# plot test-loss vs no. of epoch (10-200) by training various models with different number of epochs
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
test_loss_epochs = plot_loss_vs_epochs(X_train, X_test, y_train, y_test, epochs)

# now we change the number of timesteps and n_shift and plot the test loss vs timesteps - [62(1 quarter),125(6 months),250(1 year),375(1.5 years),500(2 years)]

def plot_loss_vs_timesteps(mydata, timesteps_list, n_shift_list):
    test_loss = []
    for i in timesteps_list:
        for j in n_shift_list:
            X, y = create_input_output_pairs(mydata, i, 8, 7, j)
            X_train, X_test, y_train, y_test = make_train_test(X, y, train_percent = 0.6)
            modelT = model(X_train.shape[1], X_train.shape[2], y_train.shape[1])
            modelT.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0, callbacks=[callback])
            test_loss.append(evaluate_model(modelT,X_test, y_test))
    return test_loss

timesteps_list = [62, 125, 250, 375, 500]
n_shift_list = [10,25,50,75,100]

test_loss_timesteps = plot_loss_vs_timesteps(data, timesteps_list, n_shift_list)

# now plot a 2d heatmap of test loss vs timesteps and n_shift
test_loss_timesteps = np.array(test_loss_timesteps)
test_loss_timesteps = test_loss_timesteps.reshape(5,5)
# write x-axis as n_shift, And y-axis as various timesteps values
plt.imshow(test_loss_timesteps, cmap='hot', interpolation='nearest')
plt.show()



# Now use Linear Regression to predict the VIX

# import the required libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# make input-output pairs

def LR_create_input_output_pairs(mydata, output_timesteps, length):
    X, y = [], []
    for i in range(len(mydata) - length - output_timesteps + 1):
        X.append(mydata[i:i+length, :])
        y.append(mydata[i+length:i+length+output_timesteps, 2])
    X = np.array(X)
    Y = np.array(y)
    print(X.shape, Y.shape)
    return X, Y

# create input-output pairs
LR_X, LR_Y = LR_create_input_output_pairs(data, 3, 50)
# split the data into train and test
LR_X_train, LR_X_test, LR_Y_train, LR_Y_test = train_test_split(LR_X, LR_Y, test_size=0.2, random_state=0)

# create a linear regression model
LR_model = LinearRegression()
# train the model
LR_model.fit(LR_X_train, LR_Y_train)
# predict the output
LR_Y_pred = LR_model.predict(LR_X_test)
# calculate the mean squared error
LR_mse = mean_squared_error(LR_Y_test, LR_Y_pred)
# calculate the r2 score
LR_r2 = r2_score(LR_Y_test, LR_Y_pred)
print('Mean squared error: ', LR_mse)
print('R2 score: ', LR_r2)

# plot the predicted output vs actual output
plt.plot(LR_Y_test, label='Actual')
plt.plot(LR_Y_pred, label='Predicted')
plt.legend()
plt.show()


# Now use Random Forest Regression to predict the VIX

# import the required libraries
from sklearn.ensemble import RandomForestRegressor

# create a random forest regression model
RF_model = RandomForestRegressor(n_estimators=100, random_state=0)
# train the model
RF_model.fit(LR_X_train, LR_Y_train)
# predict the output
RF_Y_pred = RF_model.predict(LR_X_test)
# calculate the mean squared error
RF_mse = mean_squared_error(LR_Y_test, RF_Y_pred)
# calculate the r2 score
RF_r2 = r2_score(LR_Y_test, RF_Y_pred)
print('Mean squared error: ', RF_mse)
print('R2 score: ', RF_r2)

# plot the predicted output vs actual output
plt.plot(LR_Y_test, label='Actual')
plt.plot(RF_Y_pred, label='Predicted')
plt.legend()
plt.show()

# Now use Artificial Neural Network to predict the VIX

# import the required libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

# create input-output pairs
ANN_X, ANN_Y = create_input_output_pairs(data, 60, 8, 3, 10)

# convert into 1-dim
ANN_X = ANN_X.reshape(ANN_X.shape[0], ANN_X.shape[1]*ANN_X.shape[2])

# split the data into train and test
ANN_X_train, ANN_X_test, ANN_Y_train, ANN_Y_test = make_train_test(ANN_X, ANN_Y, train_percent = 0.8)

# create a sequential model
ANN_model = Sequential()
# add layers to the model and don't use LSTMs
ANN_model.add(Dense(32, input_dim=ANN_X_train.shape[1], activation='relu'))
ANN_model.add(Dense(16, activation='relu'))
ANN_model.add(Dense(8, activation='relu'))
ANN_model.add(Dense(4, activation='relu'))
ANN_model.add(Dense(ANN_Y_train.shape[1]))
# compile the model
ANN_model.compile(loss='mean_squared_error', optimizer='adam')
# train the model
ANN_model.fit(ANN_X_train, ANN_Y_train, epochs=100, batch_size=1, verbose=0, callbacks=[callback])
# predict the output
ANN_Y_pred = ANN_model.predict(ANN_X_test)
# calculate the mean squared error
ANN_mse = mean_squared_error(ANN_Y_test, ANN_Y_pred)
# calculate the r2 score
ANN_r2 = r2_score(ANN_Y_test, ANN_Y_pred)
print('Mean squared error: ', ANN_mse)
print('R2 score: ', ANN_r2)

# plot first arguement the predicted output vs actual output
plt.plot(ANN_Y_test[0], label='Actual')
plt.plot(ANN_Y_pred[0], label='Predicted')
plt.legend()
plt.show()

# plot second arguement the predicted output vs actual output
plt.plot(ANN_Y_test[1], label='Actual')
plt.plot(ANN_Y_pred[1], label='Predicted')
plt.legend()
plt.show()

# plot third arguement the predicted output vs actual output
plt.plot(ANN_Y_test[2], label='Actual')
plt.plot(ANN_Y_pred[2], label='Predicted')
plt.legend()
plt.show()

# now use ARIMA to predict the VIX

# import the required libraries
import statsmodels.api as sm

# create input-output pairs
ARIMA_X, ARIMA_Y = create_input_output_pairs(data, 60, 8, 3, 10)

# split the data into train and test
ARIMA_X_train, ARIMA_X_test, ARIMA_Y_train, ARIMA_Y_test = make_train_test(ARIMA_X, ARIMA_Y, train_percent = 0.8)

# create a ARIMA model
ARIMA_model = sm.tsa.arima.ARIMA(ARIMA_X_train, order=(1,1,1))
ARIMA_model_fit = ARIMA_model.fit()

# predict the output
ARIMA_Y_pred = ARIMA_model_fit.forecast(steps=ARIMA_X_test.shape[0])[0]
# calculate the mean squared error
ARIMA_mse = mean_squared_error(ARIMA_Y_test, ARIMA_Y_pred)
# calculate the r2 score
ARIMA_r2 = r2_score(ARIMA_Y_test, ARIMA_Y_pred)
print('Mean squared error: ', ARIMA_mse)
print('R2 score: ', ARIMA_r2)

# plot the predicted output vs actual output
plt.plot(ARIMA_Y_test, label='Actual')
plt.plot(ARIMA_Y_pred, label='Predicted')
plt.legend()
plt.show()
