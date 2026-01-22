import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_ticket_1770(self):
    """Should not segfault on python 3k"""
    import numpy as np
    try:
        a = np.zeros((1,), dtype=[('f1', 'f')])
        a['f1'] = 1
        a['f2'] = 1
    except ValueError:
        pass
    except Exception:
        raise AssertionError