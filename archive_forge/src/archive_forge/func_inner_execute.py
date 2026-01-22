import inspect
import warnings
from functools import wraps, partial
from typing import Callable, Sequence, Optional, Union, Tuple
import logging
from cachetools import LRUCache, Cache
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import ResultBatch
from .set_shots import set_shots
from .jacobian_products import (
def inner_execute(tapes: Sequence[QuantumTape], **_) -> ResultBatch:
    """Execution that occurs within a machine learning framework boundary.

        Closure Variables:
            expand_fn (Callable[[QuantumTape], QuantumTape]): A device preprocessing step
            numpy_only (bool): whether or not to convert the data to numpy or leave as is
            cached_device_execution (Callable[[Sequence[QuantumTape]], ResultBatch])

        """
    if expand_fn:
        tapes = tuple((expand_fn(t) for t in tapes))
    if numpy_only:
        tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in tapes))
    return cached_device_execution(tapes)