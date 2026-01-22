from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated
@property
def min_absolute_change(self):
    """
        Value of the minimum absolute change required to pass model comparison with baseline model.
        """
    return self._min_absolute_change