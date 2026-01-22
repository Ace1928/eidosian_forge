from typing import Dict, Any, Optional, TYPE_CHECKING
import sklearn.datasets
import sklearn.metrics
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.core import Booster
import pickle
import ray
from ray import train, tune
from ray.tune.schedulers import ResourceChangingScheduler, ASHAScheduler
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment import Trial
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
def get_best_model_checkpoint(best_result: 'ray.train.Result'):
    best_bst = xgb.Booster()
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        to_load = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)
        if not os.path.exists(to_load):
            with open(os.path.join(checkpoint_dir, 'checkpoint'), 'rb') as f:
                _, _, raw_model = pickle.load(f)
            to_load = bytearray(raw_model)
        best_bst.load_model(to_load)
    accuracy = 1.0 - best_result.metrics['eval-logloss']
    print(f'Best model parameters: {best_result.config}')
    print(f'Best model total accuracy: {accuracy:.4f}')
    return best_bst