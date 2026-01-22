import numpy as np
from onnx.reference.op_run import OpRun
def reshape_reference_implementation(data: np.ndarray, shape: np.ndarray, allowzero: int=0) -> np.ndarray:
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped