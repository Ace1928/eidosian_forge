import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def verify_operation(function):
    """
    Verifies that `tensor` is the same shape across all processes. Only ran if `PartialState().debug` is `True`.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        if PartialState().distributed_type == DistributedType.NO or not PartialState().debug:
            return function(*args, **kwargs)
        operation = f'{function.__module__}.{function.__name__}'
        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            tensor = args[0]
        if PartialState().device.type != find_device(tensor).type:
            raise DistributedOperationException(f'One or more of the tensors passed to {operation} were not on the {tensor.device.type} while the `Accelerator` is configured for {PartialState().device.type}. Please move it to the {PartialState().device.type} before calling {operation}.')
        shapes = get_shape(tensor)
        output = gather_object([shapes])
        if output[0] is not None:
            are_same = output.count(output[0]) == len(output)
            if not are_same:
                process_shape_str = '\n  - '.join([f'Process {i}: {shape}' for i, shape in enumerate(output)])
                raise DistributedOperationException(f'Cannot apply desired operation due to shape mismatches. All shapes across devices must be valid.\n\nOperation: `{operation}`\nInput shapes:\n  - {process_shape_str}')
        return function(*args, **kwargs)
    return wrapper