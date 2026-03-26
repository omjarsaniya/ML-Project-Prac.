import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingition import DataIngestion
from src.mlproject.components.data_ingition import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.components.model_trainer import ModelTrainerConfig




import dagshub # type: ignore
dagshub.init(repo_owner="ombjarsaniya123", repo_name="ML-Project-Prac")

import mlflow

with mlflow.start_run():
    mlflow.log_params({"parameters": 10})
    





if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)

        #model training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_array, test_array))

    except Exception as e:
        logging.info("An exception has occurred")
        raise CustomException(e, sys)
    
    

    
