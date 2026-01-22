import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
@property
def original_device(self):
    """Return the original device."""
    return self._original_device