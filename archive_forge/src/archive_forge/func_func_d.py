import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
@doc(func_a, method='func_d', operation='D')
def func_d(whatever):
    pass