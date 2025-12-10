import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object

import sys
import os
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dataset Into Dependant And Independent Features')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            ## Train Multiple Models

            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
            }

            models_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f'Models Report \n{models_report}')

            best_model_name = [k for k ,v in models_report.items() if v == max(list(models_report.values()))][0]
            best_model = models[best_model_name]

            logging.info(f'Best Model Is {best_model_name} And Its Score Is {max(list(models_report.values()))}')
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)

        except Exception as e:
            logging.info("Exception Occured In Model Training")
            raise CustomException (e, sys)

