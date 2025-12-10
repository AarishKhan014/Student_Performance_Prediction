import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
import pickle
from sklearn.metrics import r2_score


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
    


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            
            ## Make Predictions

            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = score

        return report
    
    except Exception as e:
        logging.info('Exception Occured in Model Evaluation')
        raise CustomException(e, sys)