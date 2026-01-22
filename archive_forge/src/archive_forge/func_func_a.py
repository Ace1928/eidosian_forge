import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
@doc(method='func_a', operation='A')
def func_a(whatever):
    """
    This is the {method} method.

    It computes {operation}.
    """
    pass