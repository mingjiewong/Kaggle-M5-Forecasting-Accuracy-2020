import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

class Load:
    #To reduce memory usage
    def __init__(self,train_sales='',calendar=''):
        self.train_sales = pd.read_csv(train_sales)
        self.calendar = pd.read_csv(calendar)
        self.float_cols = [c for c in self.train_sales if self.train_sales[c].dtype == "float64"]
        self.int_cols = [c for c in self.train_sales if self.train_sales[c].dtype in ["int64","int32"]]

    def downcast_dtypes(self):
        self.train_sales[self.float_cols] = self.train_sales[self.float_cols].astype(np.float32)
        self.train_sales[self.int_cols] = self.train_sales[self.int_cols].astype(np.int16)
        return self.train_sales

class Preprocess:
    # Preprocess: remove id, item_id, dept_id, cat_id, store_id, state_id columns
    def __init__(self,loaded_train_sales,loaded_calendar,startDay=350):
        # Remove the first 350 days in train sales data due to zero_inflated data
        self.loaded_train_sales = loaded_train_sales.T[6 + startDay:]
        self.calendar = loaded_calendar

        # Initialize a dataframe with zeros for 1969 days in the calendar
        self.daysBeforeEvent1 = pd.DataFrame(np.zeros((1969,1)))
        self.daysBeforeEvent2 = pd.DataFrame(np.zeros((1969,1)))
        self.snap_CA = pd.DataFrame(np.zeros((1969,1)))
        self.snap_TX = pd.DataFrame(np.zeros((1969,1)))
        self.snap_WI = pd.DataFrame(np.zeros((1969,1)))

    def label_calendar(self):
        for x,y in self.calendar.iterrows():
            if((pd.isnull(self.calendar["event_name_1"][x])) == False):
                self.daysBeforeEvent1[0][x-1] = 1

            if((pd.isnull(self.calendar["event_name_2"][x])) == False):
                self.daysBeforeEvent2[0][x-1] = 1

            if((pd.isnull(self.calendar["snap_CA"][x])) == False):
                self.snap_CA[0][x] = 1

            if((pd.isnull(self.calendar["snap_TX"][x])) == False):
                self.snap_TX[0][x] = 1

            if((pd.isnull(self.calendar["snap_WI"][x])) == False):
                self.snap_WI[0][x] = 1

        return self.daysBeforeEvent1, self.daysBeforeEvent2, self.snap_CA, self.snap_TX, self.snap_WI

class SplitDataset:
    # split dataset into evaluation (last 2 weeks), validation (first 2 weeks), training
    def __init__(self, loaded_train_sales,
                 daysBeforeEvent1, daysBeforeEvent2,
                 snap_CA, snap_TX, snap_WI, startDay=350):
        # Remove the first 350 days in train sales data due to zero_inflated data
        self.loaded_train_sales = loaded_train_sales

        # input for predicting validation period day 1941 to 1969
        self.daysBeforeEvent1_eval = daysBeforeEvent1[1941:]
        self.daysBeforeEvent2_eval = daysBeforeEvent2[1941:]
        self.snap_CA_eval = snap_CA[1941:]
        self.snap_TX_eval = snap_TX[1941:]
        self.snap_WI_eval = snap_WI[1941:]

        # input for predicting validation period day 1913 to 1941
        self.daysBeforeEvent1_valid = daysBeforeEvent1[1913:1941]
        self.daysBeforeEvent2_valid = daysBeforeEvent2[1913:1941]
        self.snap_CA_valid = snap_CA[1913:1941]
        self.snap_TX_valid = snap_TX[1913:1941]
        self.snap_WI_valid = snap_WI[1913:1941]

        # input for training as a feature
        self.daysBeforeEvent1_train = daysBeforeEvent1[startDay:1941]
        self.daysBeforeEvent2_train = daysBeforeEvent2[startDay:1941]
        self.snap_CA_train = snap_CA[startDay:1941]
        self.snap_TX_train = snap_TX[startDay:1941]
        self.snap_WI_train = snap_WI[startDay:1941]

    def concatenate(self):
        #Before concatanation with our main data "dt", indexes are made same and column name is changed to "oneDayBeforeEvent"
        self.daysBeforeEvent1_train.columns = ["oneDayBeforeEvent1"]
        self.daysBeforeEvent1_train.index = self.loaded_train_sales.index

        self.daysBeforeEvent2_train.columns = ["oneDayBeforeEvent2"]
        self.daysBeforeEvent2_train.index = self.loaded_train_sales.index

        self.snap_CA_train.columns = ["snap_CA"]
        self.snap_CA_train.index = self.loaded_train_sales.index

        self.snap_TX_train.columns = ["snap_TX"]
        self.snap_TX_train.index = self.loaded_train_sales.index

        self.snap_WI_train.columns = ["snap_WI"]
        self.snap_WI_train.index = self.loaded_train_sales.index

        self.concat_train_sales = pd.concat([self.loaded_train_sales, self.daysBeforeEvent1_train,
                                             self.daysBeforeEvent2_train, self.snap_CA_train,
                                             self.snap_TX_train, self.snap_WI_train], axis = 1, sort=False)

        return self.concat_train_sales

class ScalingTrainSales:
    # Feature Scaling: Scale features using min-max scaler in range 0-1
    def __init__(self,concat_train_sales,timesteps=7,feature_range=(0,1),startDay=350):
        self.concat_train_sales = concat_train_sales
        self.timesteps = timesteps
        self.feature_range = feature_range
        self.X_train = []
        self.y_train = []
        self.startDay = startDay

    def gen_train_data(self):
        #Minmax Scaling
        sc = MinMaxScaler(feature_range=self.feature_range)
        train_sales_scaled = sc.fit_transform(self.concat_train_sales)

        for i in range(self.timesteps, 1941 - self.startDay):
            self.X_train.append(train_sales_scaled[i-self.timesteps:i])
            self.y_train.append(train_sales_scaled[i][0:30490])

        #Convert to np array to be able to feed the LSTM model
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        return self.X_train, self.y_train, sc
