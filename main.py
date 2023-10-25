from src import data_ingestion, data_transformation, model_trainer, model_predictor
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd

if __name__ == "__main__":
    train_df, test_df = data_ingestion.DataIngestion.load_data()
    x = [train_df, test_df]
    for i in x:
        i = data_transformation.ExtractFeatures.time_process(i)
        i = model_trainer.PreProcessModel.wind_0_fill(i)
    train_df, test_df = data_transformation.ExtractFeatures.add_holiday(train_df, test_df)
    for i in x:
        i = data_transformation.ExtractFeatures.extract_peak_feature(i)    
        i = data_transformation.ExtractFeatures.name_process(i)
    train_df = pd.get_dummies(train_df,columns=['season2','weather2'])
    test_df = pd.get_dummies(test_df,columns=['season2','weather2'])
    RFR_X_train, GBR_X_train = data_transformation.PreProcessData.feature_selection(train_df)
    RFR_X_test, GBR_X_test = data_transformation.PreProcessData.feature_selection(test_df)

    y_casual, y_registered, y_count = data_transformation.PreProcessData.target_preprocessing(train_df)

    model_trainer.ModelTrainer.RFR_model(RFR_X_train, y_casual, y_registered, y_count)
    model_trainer.ModelTrainer.GBR_model(GBR_X_train, y_casual, y_registered, y_count)

    y_pred1, y_pred2 = model_predictor.Model_Predictor.make_predictions(RFR_X_test, GBR_X_test)
    model_predictor.Model_Predictor.store_predictions(test_df, y_pred1, 1)
    model_predictor.Model_Predictor.store_predictions(test_df, y_pred2, 2)
