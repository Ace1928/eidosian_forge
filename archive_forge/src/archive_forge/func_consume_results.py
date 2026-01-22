import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def consume_results(self, bitstrings):
    """Add bitstrings sampled according to `meas_spec`.

        We don't validate that bitstrings were sampled correctly according
        to `meas_spec` (how could we?) so please be careful. Consider
        using `measure_observables` rather than calling this method yourself.
        """
    if bitstrings.dtype != np.uint8:
        raise ValueError('`bitstrings` should be of type np.uint8')
    self.bitstrings = np.append(self.bitstrings, bitstrings, axis=0)
    self.chunksizes = np.append(self.chunksizes, [len(bitstrings)], axis=0)
    self.timestamps = np.append(self.timestamps, [np.datetime64(datetime.datetime.now())])