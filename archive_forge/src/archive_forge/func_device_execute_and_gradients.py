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
def device_execute_and_gradients(internal_tapes, **gradient_kwargs):
    numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in internal_tapes))
    return set_shots(device, override_shots)(device.execute_and_gradients)(numpy_tapes, **gradient_kwargs)