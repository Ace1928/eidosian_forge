import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_importer(self):
    attempted_import = []

    def _importer():
        attempted_import.append(True)
        return attempt_import('pyomo.common.tests.dep_mod', defer_check=False)[0]
    mod, avail = attempt_import('foo', importer=_importer, defer_check=True)
    self.assertEqual(attempted_import, [])
    self.assertIsInstance(mod, DeferredImportModule)
    self.assertTrue(avail)
    self.assertEqual(attempted_import, [True])
    self.assertIs(mod._indicator_flag._module, dep_mod)