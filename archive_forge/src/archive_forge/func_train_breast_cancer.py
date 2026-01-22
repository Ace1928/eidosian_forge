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
def train_breast_cancer(config: dict):
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    xgb_model = None
    checkpoint = train.get_checkpoint()
    if checkpoint:
        xgb_model = xgb.Booster()
        with checkpoint.as_directory() as checkpoint_dir:
            xgb_model.load_model(os.path.join(checkpoint_dir, CHECKPOINT_FILENAME))
    config['nthread'] = int(train.get_context().get_trial_resources().head_cpus)
    print(f'nthreads: {config['nthread']} xgb_model: {xgb_model}')
    xgb.train(config, train_set, evals=[(test_set, 'eval')], verbose_eval=False, xgb_model=xgb_model, callbacks=[TuneReportCheckpointCallback(filename=CHECKPOINT_FILENAME, frequency=1)])