from pickle import load
from src import config_entity
config = config_entity.ConfigFile.parse_config(config_entity.CONFIG_FILE)
from pathlib import Path
import pandas as pd

import numpy as np

class Model_Predictor:
    def make_predictions(RFR_X_test, GBR_X_test):
        RFR_model1_path = Path(config["model_trainer"]["RFR_model1_path"])
        trained_model_RFR1 = load(open(RFR_model1_path, "rb"))
        RFR_pre_casual = trained_model_RFR1.predict(RFR_X_test)
        RFR_pre_casual=np.exp(RFR_pre_casual)-1
        RFR_model2_path = Path(config["model_trainer"]["RFR_model2_path"])
        trained_model_RFR2 = load(open(RFR_model2_path, "rb"))
        RFR_pre_registered = trained_model_RFR2.predict(RFR_X_test)
        RFR_pre_registered=np.exp(RFR_pre_registered)-1
        RFR_pred = RFR_pre_casual+RFR_pre_registered

        GBR_model1_path = Path(config["model_trainer"]["GBR_model1_path"])
        trained_model_GBR1 = load(open(GBR_model1_path, "rb"))
        GBR_pre_casual = trained_model_GBR1.predict(GBR_X_test)
        GBR_pre_casual=np.exp(GBR_pre_casual)-1
        GBR_model2_path = Path(config["model_trainer"]["GBR_model2_path"])
        trained_model_GBR2 = load(open(GBR_model2_path, "rb"))
        GBR_pre_registered = trained_model_GBR2.predict(GBR_X_test)
        GBR_pre_registered=np.exp(GBR_pre_registered)-1
        GBR_pred = GBR_pre_casual+GBR_pre_registered
        y_pred1 = 0.2 * RFR_pred + 0.8 * GBR_pred

        RFR_model3_path = Path(config["model_trainer"]["RFR_model3_path"])
        trained_model_RFR3 = load(open(RFR_model3_path, "rb"))
        RFR_pre_count = trained_model_RFR3  .predict(RFR_X_test)
        RFR_pre_count = np.exp(RFR_pre_count)-1

        GBR_model3_path = Path(config["model_trainer"]["GBR_model3_path"])
        trained_model_GBR3 = load(open(GBR_model3_path, "rb"))
        GBR_pre_count = trained_model_GBR3.predict(GBR_X_test)
        GBR_pre_count = np.exp(GBR_pre_count)-1
        y_pred2 = 0.2 * RFR_pre_count + 0.8 * GBR_pre_count
        return y_pred1, y_pred2

    def store_predictions(df, predictions, count):
        submissions_csv_path = Path(config["model_predictor"]["submissions_csv_path"])
        submissions_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if(count == 1):
            submit1 = pd.DataFrame({'datetime':df['datetime'],'count':predictions})
            submit1.to_csv('submissions/submisssion_1.csv',index=False)
        else:
            submit2 = pd.DataFrame({'datetime':df['datetime'],'count':predictions})
            submit2.to_csv('submissions/submisssion_2.csv',index=False)