import logging
import os
from datetime import datetime


## Create Logs Directory
logs_dir = os.join.path(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

## Create TimeStamp File
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H:%M:%S')}.log"
LOG_FILE_PATH = os.join.path(logs_dir, LOG_FILE)

## Configure Logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)