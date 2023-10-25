import yaml

CONFIG_FILE = "src/config.yml"

class ConfigFile:
    def parse_config(CONFIG_FILE):
        with open(CONFIG_FILE, "rb") as f:
            config = yaml.safe_load(f)
        return config