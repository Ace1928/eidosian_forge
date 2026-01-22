from pyomo.common.gc_manager import PauseGC
import gc
import pyomo.common.unittest as unittest
def test_gc_disable(self):
    self.assertTrue(gc.isenabled())
    pgc = PauseGC()
    self.assertTrue(gc.isenabled())
    pgc.close()
    self.assertTrue(gc.isenabled())
    with pgc:
        self.assertFalse(gc.isenabled())
    self.assertTrue(gc.isenabled())