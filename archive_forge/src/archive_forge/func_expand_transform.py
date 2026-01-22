import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
@property
def expand_transform(self):
    """The expand transform."""
    return self._expand_transform