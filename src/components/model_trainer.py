import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.mode_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training and test input data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor(),
            }
            model_report:dict=evaluate_models(X_train=X_train, 
                               y_train=Y_train, 
                               models=models, 
                               cv=5)
                ## getting the best model from the dictionary 
            best_model_score = max(sorted(model_report.values()))
            ## to get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on cross validation process")
            best_model.fit(X_train, Y_train)
            save_object(file_path=self.mode_trainer_config.train_model_file_path,
                        obj=best_model)
            logging.info("we Saved and trained the best model")
            y_pred = best_model.predict(X_test)
            r2_square = r2_score(Y_test, y_pred)
            return f"the R2 for {best_model_name} is {r2_square} and this is the best model !!!"
        
        except Exception as e:
            raise CustomException(e, sys)
