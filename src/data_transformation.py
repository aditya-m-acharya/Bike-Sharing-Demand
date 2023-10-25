import pandas as pd
import numpy as np
from src import config_entity

class ExtractFeatures:
    def time_process(df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.day_of_week
        df['hour'] = df['datetime'].dt.hour
        df['week'] = df['datetime'].dt.weekofyear
        return df
    
    def get_day(day_start):
        day_end = day_start + pd.offsets.DateOffset(hours=23)
        return pd.date_range(day_start, day_end, freq="H")

    def add_holiday(train, test):
        dt = pd.DatetimeIndex(train['datetime'])
        train.set_index(dt, inplace=True)
        dtt = pd.DatetimeIndex(test['datetime'])
        test.set_index(dtt, inplace=True)
        train.loc[ExtractFeatures.get_day(pd.datetime(2011, 4, 15)), "workingday"] = 1
        train.loc[ExtractFeatures.get_day(pd.datetime(2012, 4, 16)), "workingday"] = 1
        test.loc[ExtractFeatures.get_day(pd.datetime(2011, 11, 25)), "workingday"] = 0
        test.loc[ExtractFeatures.get_day(pd.datetime(2012, 11, 23)), "workingday"] = 0
        test.loc[ExtractFeatures.get_day(pd.datetime(2011, 12, 24)), "workingday"] = 0
        test.loc[ExtractFeatures.get_day(pd.datetime(2011, 12, 31)), "workingday"] = 0
        test.loc[ExtractFeatures.get_day(pd.datetime(2012, 12, 26)), "workingday"] = 0
        test.loc[ExtractFeatures.get_day(pd.datetime(2012, 12, 31)), "workingday"] = 0
        train.loc[ExtractFeatures.get_day(pd.datetime(2011, 4, 15)), "holiday"] = 0
        train.loc[ExtractFeatures.get_day(pd.datetime(2012, 4, 16)), "holiday"] = 0
        test.loc[ExtractFeatures.get_day(pd.datetime(2011, 11, 25)), "holiday"] = 1
        test.loc[ExtractFeatures.get_day(pd.datetime(2012, 11, 23)), "holiday"] = 1
        test.loc[ExtractFeatures.get_day(pd.datetime(2011, 12, 24)), "holiday"] = 1
        test.loc[ExtractFeatures.get_day(pd.datetime(2011, 12, 31)), "holiday"] = 1
        test.loc[ExtractFeatures.get_day(pd.datetime(2012, 12, 31)), "holiday"] = 1
        test.loc[ExtractFeatures.get_day(pd.datetime(2012, 5, 21)), "holiday"] = 1
        train.loc[ExtractFeatures.get_day(pd.datetime(2012, 6, 1)), "holiday"] = 1
        train.reset_index(inplace=True, drop=True)
        test.reset_index(inplace=True, drop=True)
        return train, test

    def name_process(df):
        df['season2'] = df['season']
        df['weather2'] = df['weather']
        df['season2'] = df['season2'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})
        df['weather2'] = df['weather2'].map({1:'Clear',2:'Mist',3:'Light_Snow',4:'Heavy_Rain'})
    #     df['month'] = df['month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})   
        return df
    def ohe(df):    
        df = pd.get_dummies(df,columns=['season2'])
        #df=pd.get_dummies(df,columns=['weather2'])
        return df
    
    def extract_peak_feature(df):
        df['peak'] = df[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 12)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)
        return df
    
class PreProcessData:
    def feature_selection(df):
        config = config_entity.ConfigFile.parse_config(config_entity.CONFIG_FILE)
        '''Define columns for each model. (Hardcode here or take from config?).'''
        #All_feature_columns = list(config["data_transformation"]["All_feature_columns"])
        RFR_feature_columns = list(config["data_transformation"]["RFR_feature_columns"])
        GBR_feature_columns = list(config["data_transformation"]["GBR_feature_columns"])
        return df[RFR_feature_columns].values, df[GBR_feature_columns].values

    def target_preprocessing(df):
        '''Take log of target column'''
        y_casual = df['casual'].apply(lambda x: np.log1p(x)).values
        y_registered = df['registered'].apply(lambda x: np.log1p(x)).values
        y_count = df['count'].apply(lambda x: np.log1p(x)).values
        return y_casual, y_registered, y_count