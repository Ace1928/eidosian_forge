from typing import Dict, List
import sklearn.datasets
import sklearn.metrics
import os
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
def train_breast_cancer_cv(config: dict):
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)

    def average_cv_folds(results_dict: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in results_dict.items()}
    train_set = xgb.DMatrix(data, label=labels)
    xgb.cv(config, train_set, verbose_eval=False, stratified=True, callbacks=[TuneReportCheckpointCallback(results_postprocessing_fn=average_cv_folds, frequency=0)])