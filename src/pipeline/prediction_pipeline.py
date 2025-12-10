import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object




class PredictPipeline:
    def __init__(self):
        pass


    
    def prediction(self, features):
        try:
            scaler_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            scaler = load_object(scaler_path)
            model = load_object(model_path)

            data_scaled = scaler.transform(features)
            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info("Exception Occured in Prediction Pipeline")
            raise CustomException (e, sys)
        
    

class StoringData:
    def __init__ (self,
                  hours_studied,
                  previous_scores,
                  extracurricular_activities,
                  sleep_hours,
                  sample_question_papers_practiced):
        

        self.hours_studied = hours_studied
        self.previous_scores = previous_scores
        self.extracurricular_activities = extracurricular_activities
        self.sleep_hours = sleep_hours
        self.sample_question_papers_practiced = sample_question_papers_practiced


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Hours Studied':self.hours_studied,
                'Previous Scores':self.previous_scores,
                'Extracurricular Activities':self.extracurricular_activities,
                'Sleep Hours':self.sleep_hours,
                'Sample Question Papers Practiced':self.sample_question_papers_practiced
            }

            if custom_data_input_dict['Extracurricular Activities'] == 'Yes':
                custom_data_input_dict['Extracurricular Activities'] = 1


            df = pd.DataFrame(custom_data_input_dict, index=[0])
            logging.info('Data Collected And DataFrame Created.')
            return df
        
        except Exception as e:
            logging.info("Exception Occured in Prediction Pipeline.")
            raise CustomException(e, sys)
