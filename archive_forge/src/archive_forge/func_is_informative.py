import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
@property
def is_informative(self):
    """``True`` if the transform is informative."""
    return self._is_informative