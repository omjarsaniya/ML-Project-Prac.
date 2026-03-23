import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingition import DataIngestion
from src.mlproject.components.data_ingition import DataIngestionConfig


if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
    except Exception as e:
        logging.info("An exception has occurred")
        raise CustomException(e, sys)
    
    

    
