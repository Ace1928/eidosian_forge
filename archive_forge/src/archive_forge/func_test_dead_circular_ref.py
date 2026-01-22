import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_dead_circular_ref(self):
    o = weakref.ref(greenlet.greenlet(greenlet.getcurrent).switch())
    gc.collect()
    if o() is not None:
        import sys
        print('O IS NOT NONE.', sys.getrefcount(o()))
    self.assertIsNone(o())
    self.assertFalse(gc.garbage, gc.garbage)