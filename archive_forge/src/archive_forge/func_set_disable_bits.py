from __future__ import annotations
import warnings
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from enum import Enum
import numpy as np
from qiskit import circuit
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawings, types
from qiskit.visualization.timeline.stylesheet import QiskitTimelineStyle
def set_disable_bits(self, bit: types.Bits, remove: bool=True):
    """Interface method to control visibility of bits.

        Specified object in the blocked list will not be shown.

        Args:
            bit: A qubit or classical bit object to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
    if remove:
        self.disable_bits.add(bit)
    else:
        self.disable_bits.discard(bit)