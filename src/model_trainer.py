from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from src import config_entity
config = config_entity.ConfigFile.parse_config(config_entity.CONFIG_FILE)
from pathlib import Path

# from cloudpickle import dump
from pickle import dump

class PreProcessModel:
    def wind_0_fill(df):
        wind_0 = df[df['windspeed']==0]
        wind_not0 = df[df['windspeed']!=0]
        y_label = wind_not0['windspeed']

        clf = RandomForestClassifier(n_estimators=1000,max_depth=10,random_state=0)
        windcolunms = ['season', 'weather', 'temp', 'atemp', 'humidity', 'hour', 'month']
        clf.fit(wind_not0[windcolunms], y_label.astype('int'))
        pred_y = clf.predict(wind_0[windcolunms])

        wind_0['windspeed'] = pred_y
        df_rfw = wind_not0.append(wind_0)
        df_rfw.reset_index(inplace=True, drop=True)
        return df_rfw

class ModelTrainer:
    def RFR_model(RFR_X_train, y_casual, y_registered, y_count):
        params = {'n_estimators': 1000, 
                'max_depth': 15, 
                'random_state': 0, 
                'min_samples_split' : 5, 
                'n_jobs': -1}

        RFR1 = RandomForestRegressor(**params)
        RFR1.fit(RFR_X_train,y_casual)
        RFR_model1_path = Path(config["model_trainer"]["RFR_model1_path"])
        RFR_model1_path.parent.mkdir(parents=True, exist_ok=True)
        dump(RFR1, open(RFR_model1_path, "wb"))

        RFR2 = RandomForestRegressor(**params)
        RFR2.fit(RFR_X_train,y_registered)
        RFR_model2_path = Path(config["model_trainer"]["RFR_model2_path"])
        dump(RFR2, open(RFR_model2_path, "wb"))
        
        RFR3 = RandomForestRegressor(**params)
        RFR3.fit(RFR_X_train,y_count)
        RFR_model3_path = Path(config["model_trainer"]["RFR_model3_path"])
        dump(RFR3, open(RFR_model3_path, "wb"))


    
    def GBR_model(GBR_X_train, y_casual, y_registered, y_count):
        params2 = {'n_estimators': 150, 
                'max_depth': 5, 
                'random_state': 0, 
                'min_samples_leaf' : 10, 
                'learning_rate': 0.1, 
                'subsample': 0.7}

        GBR1 = GradientBoostingRegressor(**params2)
        GBR1.fit(GBR_X_train,y_casual)
        GBR_model1_path = Path(config["model_trainer"]["GBR_model1_path"])
        dump(GBR1, open(GBR_model1_path, "wb"))

        GBR2 = GradientBoostingRegressor(**params2)
        GBR2.fit(GBR_X_train,y_registered)
        GBR_model2_path = Path(config["model_trainer"]["GBR_model2_path"])
        dump(GBR2, open(GBR_model2_path, "wb"))

        GBR3 = GradientBoostingRegressor(**params2)
        GBR3.fit(GBR_X_train,y_count)
        GBR_model3_path = Path(config["model_trainer"]["GBR_model3_path"])
        dump(GBR3, open(GBR_model3_path, "wb"))