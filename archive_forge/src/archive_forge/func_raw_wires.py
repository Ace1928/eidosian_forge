import copy
import functools
from warnings import warn
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Tuple, Optional, Union
import pennylane as qml
from pennylane.operation import Operator, DecompositionUndefinedError, EigvalsUndefinedError
from pennylane.pytrees import register_pytree
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .shots import Shots
from a classical shadow measurement"""
@property
def raw_wires(self):
    """The wires the measurement process acts on.

        For measurements involving more than one set of wires (such as
        mutual information), this is a list of the Wires objects. Otherwise,
        this is the same as :func:`~.MeasurementProcess.wires`
        """
    return self._wires