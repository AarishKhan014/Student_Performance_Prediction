import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion Process")

        try:
            #Read Dataset
            csv_path = os.path.join('notebooks', 'data', 'Student_Performance.csv')
            
            if not os.path.exists (csv_path):
                logging.info('Performance File Not Found')
                raise CustomException(f'File Not Found {csv_path}', sys)
            
            df = pd.read_csv(csv_path)
            logging.info (f'Dataset Read Successfully With Shape {df.shape}')

            ## Create Artifacts Folder If It Does'nt Exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            ## Save Raw File
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f'Raw Data Saved At {self.ingestion_config.raw_data_path}')

            ##Split Train And Test
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=786)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f'Train Data Saved At {self.ingestion_config.train_data_path}')
            logging.info(f'Test Data Saved At {self.ingestion_config.test_data_path}')

            logging.info('Data Ingestion Completed Successfully')
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path


        except Exception as e:
            logging.info ("Exception Occured In Data Ingestion")
            raise CustomException(e, sys)
        
## For Testing
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)