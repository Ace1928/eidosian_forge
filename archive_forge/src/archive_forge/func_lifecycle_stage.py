from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import ExperimentTag as ProtoExperimentTag
@property
def lifecycle_stage(self):
    """Lifecycle stage of the experiment. Can either be 'active' or 'deleted'."""
    return self._lifecycle_stage