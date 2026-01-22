import math as _math
import time as _time
import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.api import runtime
@property
def cpu_times(self) -> _numpy.ndarray:
    """A :class:`numpy.ndarray` of shape ``(n_repeat,)``, holding times spent
        on CPU in seconds.

        These values are delta of the host-side performance counter
        (:func:`time.perf_counter`) between each repeat step.
        """
    return self._ts[0]