import os 
import sys 
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import cross_val_score
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_object:
            dill.dump(obj, file_object)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, models, cv=5):
    """
    Returns a dictionary of models and their cross-validation mean R² scores.
    """
    try:
        report = {}
        for name, model in models.items():
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="r2"
            )

            cv_mean = np.mean(cv_scores)
            report[name] = cv_mean   # store only the CV mean score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)