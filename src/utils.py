import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
import pickle


def remove_correlated_features(df_corr, thresh):
    try:
        coll = set()

        for i in range(len(df_corr.columns)):
            for j in range(0, i):
                if abs(df_corr.iloc[i, j]) >= thresh:
                    coll.add(df_corr.columns[i])

        return coll

    except Exception as e:
        logging.info('Exception Occured In Removing Collinear Features.')
        raise CustomException(e, sys)




def save_object (file_path, obj):
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info('Exception Occured in Saving Object.')
        raise CustomException (e, sys)