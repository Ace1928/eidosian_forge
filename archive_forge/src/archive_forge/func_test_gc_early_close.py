from pyomo.common.gc_manager import PauseGC
import gc
import pyomo.common.unittest as unittest
def test_gc_early_close(self):
    pgc = PauseGC()
    with pgc:
        self.assertFalse(gc.isenabled())
        pgc.close()
        self.assertTrue(gc.isenabled())
    self.assertTrue(gc.isenabled())