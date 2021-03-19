import tensorflow as tf
import pandas as pd
import numpy as np
from tf.keras.models import Sequential
from tf.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(51)
np.random.seed(51)

class Config(object):
    def __init__(self):
        """
        Load CNN-LSTM model parameters.

        Attributes:
          EPOCHS (int): number of epochs
          BATCH_SIZE (int): batch size
          optimizer (obj): optimizer
          metrics (obj): metrics for optimizer
          loss (str): type of loss function
        """
        self.EPOCHS = 10
        self.BATCH_SIZE = 100

        # Optimizer Config
        self.optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
        self.metrics = tf.keras.metrics.RootMeanSquaredError()
        self.loss = 'mse'

class CNNLSTM:
    def __init__(self, X_train, y_train):
        """
        Load the training and test inputs and input shape of training to CNN-LSTM model.

        Args:
          X_train (arr): training inputs with dimensions
            [n_timeseries , n_timesteps, n_products]
          y_train (arr): test inputs with dimensions
            [n_timeseries, n_pred_products]

        Attributes:
          X_train (arr): training inputs with dimensions
            [n_timeseries , n_timesteps, n_products]
          y_train (arr): test inputs with dimensions
            [n_timeseries, n_pred_products]
          n_timesteps (int): number of timesteps
          n_products (int): number of features
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_timesteps = X_train.shape[1]
        self.n_products = X_train.shape[2]

    def gen_model(self):
        """
        Load CNN-LSTM model configurations.
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=3,strides=1, padding="causal", activation="relu",
                                   input_shape=(self.n_timesteps, self.n_products)),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding="causal"),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(30490)
        ])

    def gen_model_summary(self):
        """
        Generate model summary.
        """
        self.model.summary()

    def compile_model(self, loss, optimizer, metrics):
        """
        Compile model.

        Args:
          loss (str): type of loss function
          optimizer (obj): optimizer
          metrics (obj): metrics for optimizer
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    def run_model(self, epochs, batch_size):
        """
        Run model.

        Args:
          epochs (int): number of epochs
          batch_size (int): batch size
        """
        self.model.fit(self.X_train, self.y_train, epochs = epochs, batch_size = batch_size)


class PredictionStep:
    def __init__(self,timesteps=7,sc=MinMaxScaler(feature_range=(0,1))):
        """
        Load the parameters for sales forecasting.

        Args:
          timesteps (int): number of timesteps
          sc (obj): scaler

        Attributes:
          timesteps (int): number of timesteps
          sc (obj): scaler
        """
        self.timesteps = timesteps
        self.sc = sc

    def run_prediction(self,concat_train_sales,daysBeforeEvent1_valid,daysBeforeEvent2_valid,
                       snap_CA_valid,snap_TX_valid,snap_WI_valid):
        """
        Forecast daily sales for 28-days validation period (between day 1913 and 1941).

        Args:
          concat_train_sales (dataframe): input daily data of sales, presence of events and SNAP program
          daysBeforeEvent1_valid (dataframe): input daily data of festive events (validation)
          daysBeforeEvent2_valid (dataframe): input daily data of sporting events (validation)
          snap_CA_valid (dataframe): input daily data of SNAP program in California (validation)
          snap_TX_valid (dataframe): input daily data of SNAP program in Texas (validation)
          snap_WI_valid (dataframe): input daily data of SNAP program in Wisconsin (validation)

        Returns:
          predictions (arr): predicted sales for validation period with dimensions
            [n_valid_days]
        """
        inputs = concat_train_sales[-self.timesteps*2:-self.timesteps]
        inputs = self.sc.transform(inputs)

        X_test = []
        X_test.append(inputs[0:self.timesteps])
        X_test = np.array(X_test)
        predictions = []

        for j in range(self.timesteps,self.timesteps + 28):
            predicted_stock_price = cnn_lstm.model.predict(X_test[0,j - self.timesteps:j].reshape(1, self.timesteps, 30495))

            testInput = np.column_stack((np.array(predicted_stock_price),
                                         daysBeforeEvent1_valid.loc[1913 + j - self.timesteps],
                                         daysBeforeEvent2_valid.loc[1913 + j - self.timesteps],
                                         snap_CA_valid.loc[1913 + j - self.timesteps],
                                         snap_TX_valid.loc[1913 + j - self.timesteps],
                                         snap_WI_valid.loc[1913 + j - self.timesteps]))

            X_test = np.append(X_test, testInput).reshape(1,j + 1,30495)
            predicted_stock_price = self.sc.inverse_transform(testInput)[:,0:30490]
            predictions.append(predicted_stock_price)

        return predictions

    def run_prediction_eval(self,concat_train_sales,daysBeforeEvent1_eval,daysBeforeEvent2_eval,snap_CA_eval,
                            snap_TX_eval,snap_WI_eval):
        """
        Forecast daily sales for 28-days evaluation period (between day 1941 and 1969).

        Args:
          concat_train_sales (dataframe): input daily data of sales, presence of events and SNAP program
          daysBeforeEvent1_eval (dataframe): input daily data of festive events (evaluation)
          daysBeforeEvent2_eval (dataframe): input daily data of sporting events (evaluation)
          snap_CA_eval (dataframe): input daily data of SNAP program in California (evaluation)
          snap_TX_eval (dataframe): input daily data of SNAP program in Texas (evaluation)
          snap_WI_eval (dataframe): input daily data of SNAP program in Wisconsin (evaluation)

        Returns:
          predictions_eval (arr): predicted sales for evaluation period with dimensions
            [n_eval_days]
        """
        inputs_eval = concat_train_sales[-self.timesteps:]
        inputs_eval = self.sc.transform(inputs_eval)

        X_eval = []
        X_eval.append(inputs_eval[0:self.timesteps])
        X_eval = np.array(X_eval)
        predictions_eval = []

        for j in range(self.timesteps,self.timesteps + 28):
            predicted_stock_price = cnn_lstm.model.predict(X_eval[0,j - self.timesteps:j].reshape(1, self.timesteps, 30495))

            testInput = np.column_stack((np.array(predicted_stock_price),
                                         daysBeforeEvent1_eval.loc[1941 + j - self.timesteps],
                                         daysBeforeEvent2_eval.loc[1941 + j - self.timesteps],
                                         snap_CA_eval.loc[1941 + j - self.timesteps],
                                         snap_TX_eval.loc[1941 + j - self.timesteps],
                                         snap_WI_eval.loc[1941 + j - self.timesteps]))

            X_eval = np.append(X_eval, testInput).reshape(1,j + 1,30495)
            predicted_stock_price = self.sc.inverse_transform(testInput)[:,0:30490]
            predictions_eval.append(predicted_stock_price)

        return predictions_eval

    def gen_csv(self,predictions,predictions_eval,sample_file):
        """
        Generate the csv file for forecasted sales.

        Args:
          predictions (arr): predicted sales for validation period with dimensions
            [n_valid_days]
          predictions_eval (arr): predicted sales for evaluation period with dimensions
            [n_eval_days]
          sample_file (str): sample file path
        """
        submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))
        submission = submission.T

        submission_eval = pd.DataFrame(data=np.array(predictions_eval).reshape(28,30490))
        submission_eval = submission_eval.T

        submission = pd.concat((submission, submission_eval), ignore_index=True)

        sample_submission = pd.read_csv(sample_file)

        idColumn = sample_submission[["id"]]
        submission[["id"]] = idColumn

        cols = list(submission.columns)
        cols = cols[-1:] + cols[:-1]
        submission = submission[cols]

        colsdeneme = ["id"] + [f"F{i}" for i in range (1,29)]
        submission.columns = colsdeneme

        cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
                'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
                'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28']

        submission[cols] = submission[cols].mask(submission[cols] < 0, 0)
        submission.to_csv("submission.csv", index=False)
