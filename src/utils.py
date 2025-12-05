import os
import sys
import numpy as np
import pandas as pd
import dill
from .exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(x_train, y_train, x_test, y_test, models,param):
    try:
        report = {}

        for name, model in models.items():
            para = param.get(name,{})

            if para:
                gs = GridSearchCV(
                    estimator = model,
                    param_grid = para,
                    cv = 3,

                )
                gs.fit(x_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(x_train,y_train)
            
            models[name] = model
            
            y_train_pred = model.predict(x_train)
            y_test_pred  = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score  = r2_score(y_test,  y_test_pred)

            report[name] = {
                "train_r2": train_model_score,
                "test_r2":  test_model_score,
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)
