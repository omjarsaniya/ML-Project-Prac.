import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

from catboost import CatBoostRegressor
import mlflow
from mlflow.metrics import mae, rmse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trainded_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            parameters = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter': ['best', 'random'],
                    #'max_features': ['sqrt', 'log2', None],
                },
                "Random Forest": {
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features': ['sqrt', 'log2', None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    #'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    #'criterion': ['squared_error', 'friedman_mse'],
                    #'max_features': ['sqrt', 'log2', None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {
                    #'fit_intercept': [True, False],
                    #'normalize': [True, False],
                    #'copy_X': [True, False]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'depth': [6, 8, 10],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    #'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            } 


            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, parameters)

            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[list(models.keys())[i]] = test_model_score

            ## To get best model score from dict 
            best_model_score = max(model_report.values())

            ## To get best model name from dict
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            
            ## To get best model from dict
            best_model = models[best_model_name]

            print("This is the best model : ", best_model_name)

            model_names = list(parameters.keys())

            actual_model = ""

            for model in model_names:
                if best_model_name == model :
                    actual_model = actual_model + model

            best_parameters = parameters[actual_model]

            print("Before MLflow block")

            mlflow.set_tracking_uri("https://dagshub.com/ombjarsaniya123/ML-Project-Prac.mlflow")

            with mlflow.start_run():

                print("Before MLflow block")

                # Train best model
                best_model.fit(X_train, y_train)

                # Predict
                predicted_quality = best_model.predict(X_test)

                # Metrics
                import numpy as np

                rmse = np.sqrt(mean_squared_error(y_test, predicted_quality))
                mae = mean_absolute_error(y_test, predicted_quality)
                r2 = r2_score(y_test, predicted_quality)

                print("RMSE:", rmse)
                print("MAE:", mae)
                print("R2:", r2)

                print("Logging metrics:", rmse, mae, r2)

                # Log parameters
                for key, value in best_parameters.items():
                    mlflow.log_param(key, value)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                mlflow.sklearn.log_model(best_model, "model")



            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trainded_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        

           
    
        

