from mxnet.gluon.contrib.estimator import EpochEnd, TrainBegin, TrainEnd
from mxnet.gluon.nn import HybridSequential
import mlflow
from mlflow.utils.autologging_utils import ExceptionSafeClass, get_autologging_config
def train_end(self, estimator, *args, **kwargs):
    if isinstance(estimator.net, HybridSequential) and self.log_models:
        registered_model_name = get_autologging_config(mlflow.gluon.FLAVOR_NAME, 'registered_model_name', None)
        mlflow.gluon.log_model(estimator.net, artifact_path='model', registered_model_name=registered_model_name)