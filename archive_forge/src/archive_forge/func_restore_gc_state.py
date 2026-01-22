import contextlib
import gc
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Str, WeakRef
from traits.testing.unittest_tools import UnittestTools
@contextlib.contextmanager
def restore_gc_state():
    """Ensure that gc state is restored on exit of the with statement."""
    originally_enabled = gc.isenabled()
    try:
        yield
    finally:
        if originally_enabled:
            gc.enable()
        else:
            gc.disable()