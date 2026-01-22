import functools
import unittest.mock
import os
from typing import Callable, Type
from cirq._compat import deprecated, deprecated_class
def deprecated_cirq_ft_function() -> Callable[[Callable], Callable]:
    """Decorator to mark a function in Cirq-FT deprecated."""
    return deprecated(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)