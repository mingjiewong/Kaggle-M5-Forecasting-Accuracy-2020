from cnnlstm.preprocessing import Load, Preprocess, SplitDataset, ScalingTrainSales
from cnnlstm.forecast import Config, CNNLSTM, PredictionStep

if __name__ == '__main__':
    ### Configure file names
    train_sales_filename = '/datasets/m5-forecasting-accuracy/sales_train_evaluation.csv'
    calendar_filename = '/datasets/m5-forecasting-accuracy/calendar.csv'
    sample_file = '/datasets/m5-forecasting-accuracy/sample_submission.csv'

    ### Downcast the dataframes to reduce memory usage
    load = Load(train_sales=train_sales_filename,calendar=calendar_filename)
    train_sales = load.downcast_dtypes()
    calendar = load.calendar

    ### Add festive and sports events and SNAP programs as features to sales dataframe
    preprocessed = Preprocess(loaded_train_sales=train_sales,loaded_calendar=calendar)
    loaded_train_sales = preprocessed.loaded_train_sales
    daysBeforeEvent1, daysBeforeEvent2, snap_CA, snap_TX, snap_WI = preprocessed.label_calendar()

    ### Split the dataset into training, validation and evaluation dataset
    split = SplitDataset(loaded_train_sales, daysBeforeEvent1, daysBeforeEvent2, snap_CA, snap_TX, snap_WI)
    concat_train_sales = split.concatenate()

    ### Scale sales using Min-Max scaler to the range of 0 and 1
    scaling_train_sales = ScalingTrainSales(concat_train_sales)
    X_train, y_train, minmaxscaler = scaling_train_sales.gen_train_data()

    ### Configure the CNN-LSTM model settings
    cfg = Config()

    ### Run the CNN-LSTM model
    cnn_lstm = CNNLSTM(X_train, y_train)
    cnn_lstm.gen_model()
    cnn_lstm.compile_model(loss=cfg.loss, optimizer=cfg.optimizer, metrics=cfg.metrics)
    cnn_lstm.run_model(epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE)

    ### Forecast the sales from trained CNN-LSTM model
    pred_step = PredictionStep(sc=minmaxscaler)

    preds = pred_step.run_prediction(concat_train_sales,split.daysBeforeEvent1_valid,split.daysBeforeEvent2_valid,
                                 split.snap_CA_valid,split.snap_TX_valid,split.snap_WI_valid)

    preds_eval = pred_step.run_prediction_eval(concat_train_sales,split.daysBeforeEvent1_eval,
                                           split.daysBeforeEvent2_eval,split.snap_CA_eval,
                                           split.snap_TX_eval,split.snap_WI_eval)

    ### Generate csv file for forecasted sales
    pred_step.gen_csv(predictions=preds,predictions_eval=preds_eval,sample_file=sample_file)
