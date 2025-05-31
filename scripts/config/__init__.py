import configparser
from pathlib import Path

appconfig = configparser.ConfigParser()
config_file_path = Path(__file__).parent / 'config.ini'
appconfig.read(str(config_file_path))

# Example of how you would access values
#model_path = appconfig['Paths']['model_path']
#data_path = appconfig['Paths']['data_path']

