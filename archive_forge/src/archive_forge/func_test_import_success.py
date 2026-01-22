import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_import_success(self):
    module_obj, module_available = attempt_import('ply', 'Testing import of ply', defer_check=False)
    self.assertTrue(module_available)
    import ply
    self.assertTrue(module_obj is ply)