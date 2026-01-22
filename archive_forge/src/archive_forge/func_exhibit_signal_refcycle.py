import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
def exhibit_signal_refcycle():
    obj = set()
    signal.getsignal(signal.SIGINT)
    return weakref.ref(obj)