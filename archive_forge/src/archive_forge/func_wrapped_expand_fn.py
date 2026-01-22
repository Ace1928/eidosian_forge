from functools import wraps
import inspect
from typing import Union, Callable, Tuple
import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots
@wraps(expand_fn)
def wrapped_expand_fn(tape, *args, **kwargs):
    return ((expand_fn(tape, *args, **kwargs),), null_postprocessing)