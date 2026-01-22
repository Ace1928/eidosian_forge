from typing import Optional, TYPE_CHECKING
import numpy as np
from cirq import linalg, value
Checks if a ~= b * exp(i t) for some t.

    Args:
        actual: A numpy array.
        desired: Another numpy array.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        equal_nan: Whether or not NaN entries should be considered equal to
            other NaN entries.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error
            message.

    Raises:
        AssertionError: The matrices aren't nearly equal up to global phase.
    