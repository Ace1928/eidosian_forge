import json
import os
import warnings
import tempfile
from functools import wraps
import numpy as np
import numpy.array_api
import numpy.testing as npt
import pytest
import hypothesis
from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning
from scipy._lib import _pep440
from scipy._lib._array_api import SCIPY_ARRAY_API, SCIPY_DEVICE
def skip_if_array_api_backend(backend):

    def wrapper(func):
        reason = f'do not run with Array API backend: {backend}'
        if '.' in func.__qualname__:

            @wraps(func)
            def wrapped(self, *args, **kwargs):
                xp = kwargs['xp']
                if xp.__name__ == backend:
                    pytest.skip(reason=reason)
                return func(self, *args, **kwargs)
        else:

            @wraps(func)
            def wrapped(*args, **kwargs):
                xp = kwargs['xp']
                if xp.__name__ == backend:
                    pytest.skip(reason=reason)
                return func(*args, **kwargs)
        return wrapped
    return wrapper