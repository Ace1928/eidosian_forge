import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def convert_metric_value_to_float_if_possible(x) -> float:
    if x is None or type(x) == float:
        return x
    converter_fns_to_try = [convert_metric_value_to_float_if_ndarray, convert_metric_value_to_float_if_tensorflow_tensor, convert_metric_value_to_float_if_torch_tensor]
    for converter_fn in converter_fns_to_try:
        possible_float = converter_fn(x)
        if type(possible_float) == float:
            return possible_float
    try:
        return float(x)
    except ValueError:
        return x