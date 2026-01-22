import numpy as np
from onnx.reference.op_run import OpRun
def common_reference_implementation(data: np.ndarray, shape: np.ndarray) -> np.ndarray:
    ones = np.ones(shape, dtype=data.dtype)
    return data * ones