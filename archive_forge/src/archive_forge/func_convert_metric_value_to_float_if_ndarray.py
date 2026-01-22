import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
@_converter_requires('numpy')
def convert_metric_value_to_float_if_ndarray(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return float(_try_get_item(x))
    return x