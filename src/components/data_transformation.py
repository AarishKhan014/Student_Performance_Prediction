import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import remove_correlated_features, save_object

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')


class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()


    def get_data_transformation_object(self, train_df, target_column_name):
        try:
            logging.info('Data Transformation Initiated')
            
            temp_df = train_df.drop(columns=[target_column_name])

            categorical_columns = temp_df.columns[temp_df.dtypes == 'O']
            numerical_columns = temp_df.columns[temp_df.dtypes != 'O']


            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()) 
                ]
            )

            ## Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ('scaler', StandardScaler()) 
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logging.info('Pipeline Completed.')

            return preprocessor

        except Exception as e:
            logging.info('Exception Occured in Data Transformation Object Creation')
            raise CustomException (e, sys)
        

    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            ##Reading Train And Test Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train And Test Data Completed.")
            logging.info(f"Train DF Head: \n{train_df.head()}")
            logging.info(f"Test DF Head: \n{test_df.head()}")

            numerical_columns = train_df.columns[train_df.dtypes != 'O']
            target_column_name = 'Performance Index'

            columns_to_drop = remove_correlated_features(train_df[numerical_columns].drop(columns=[target_column_name]).corr(), 0.95)
            train_df = train_df.drop(columns=list(columns_to_drop))
            test_df = test_df.drop(columns=list(columns_to_drop))

            logging.info("Obtaining PreProcessing Data")

            preprocessing_obj = self.get_data_transformation_object(train_df, target_column_name)
            
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying Preprocessing Data On Train And Test Object')
 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            
            logging.info("Preprocessor Pickle File Saved.")

            return (
                train_arr,
                test_arr,
                preprocessing_obj
            )
            

        except Exception as e:
            logging.info("Exception Occured In Initiation Of Data Transformation")
            raise CustomException (e, sys)
