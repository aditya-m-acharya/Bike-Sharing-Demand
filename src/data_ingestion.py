from src import config_entity
from pathlib import Path
import pandas as pd

class DataIngestion:
    def load_data():
        config = config_entity.ConfigFile.parse_config(config_entity.CONFIG_FILE)
        train_data = Path(config["data_ingestion"]["train_data"])
        test_data = Path(config["data_ingestion"]["test_data"])
        train_df = pd.read_csv(train_data)
        test_df = pd.read_csv(test_data)
        return train_df, test_df