from typing import Any, List
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import insecure_hash
def hash_array(array):
    flattened_array = array.flatten()
    trimmed_array = flattened_array[0:MAX_ROWS]
    try:
        hashable_elements.append(pd.util.hash_array(trimmed_array))
    except TypeError:
        hashable_elements.append(np.int64(trimmed_array.size))
    for x in array.shape:
        hashable_elements.append(np.int64(x))