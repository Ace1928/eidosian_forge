import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_min_version(self):
    mod, avail = attempt_import('pyomo.common.tests.dep_mod', minimum_version='1.0', defer_check=False)
    self.assertTrue(avail)
    self.assertTrue(inspect.ismodule(mod))
    self.assertTrue(check_min_version(mod, '1.0'))
    self.assertFalse(check_min_version(mod, '2.0'))
    mod, avail = attempt_import('pyomo.common.tests.dep_mod', minimum_version='2.0', defer_check=False)
    self.assertFalse(avail)
    self.assertIs(type(mod), ModuleUnavailable)
    with self.assertRaisesRegex(DeferredImportError, 'The pyomo.common.tests.dep_mod module version 1.5 does not satisfy the minimum version 2.0'):
        mod.hello
    mod, avail = attempt_import('pyomo.common.tests.dep_mod', error_message='Failed import', minimum_version='2.0', defer_check=False)
    self.assertFalse(avail)
    self.assertIs(type(mod), ModuleUnavailable)
    with self.assertRaisesRegex(DeferredImportError, 'Failed import \\(version 1.5 does not satisfy the minimum version 2.0\\)'):
        mod.hello
    mod, avail = attempt_import('pyomo.common.tests.dep_mod', defer_check=True)
    self.assertTrue(check_min_version(mod, '1.0'))
    mod, avail = attempt_import('pyomo.common.tests.dep_mod', defer_check=True)
    self.assertFalse(check_min_version(mod, '2.0'))
    mod, avail = attempt_import('pyomo.common.tests.dep_mod', minimum_version='1.0')
    self.assertTrue(check_min_version(mod, '1.0'))
    mod, avail = attempt_import('pyomo.common.tests.bogus', minimum_version='1.0')
    self.assertFalse(check_min_version(mod, '1.0'))