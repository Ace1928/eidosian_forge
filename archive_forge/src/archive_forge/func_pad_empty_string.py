from typing import Union
import numpy as np
from onnx.reference.op_run import OpRun
def pad_empty_string(split_lists: Union[list, np.ndarray], padding_requirement: Union[list, int]) -> list:
    if isinstance(split_lists, list):
        return split_lists + ['' for _ in range(padding_requirement)]
    if isinstance(split_lists, np.ndarray):
        return list(map(pad_empty_string, split_lists, padding_requirement))
    raise TypeError(f"Invalid array type '{type(split_lists)}'")