import os
from sklearn.linear_model import LinearRegression
import sys

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv 
import pymysql 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


load_dotenv()
host=os.getenv('host') 
user=os.getenv('user')
password=os.getenv('password')
db=os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established successfully", mydb)
        df=pd.read_sql_query("SELECT * FROM students", mydb)
        print(df.head())

        return df
    
    except Exception as e:
        raise CustomException(e,sys)
    


def save_object(file_path, obj):
    try:
        logging.info("Saving object started")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        logging.info("Model evaluation started")
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train) 
            # train the model with best parameters found by GridSearchCV

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        
        return report
    
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)    

def load_object(file_path):
    try:
        logging.info("Loading object started")
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    
    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise CustomException(e, sys)
    


    