import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_local_deferred_import(self):
    self.assertIs(type(deps.bogus_available), DeferredImportIndicator)
    self.assertIs(type(deps.bogus), DeferredImportModule)
    if deps.bogus_available:
        self.fail('Casting bogus_available to bool returned True')
    self.assertIs(deps.bogus_available, False)
    self.assertIs(type(deps.bogus), ModuleUnavailable)
    with self.assertRaisesRegex(DeferredImportError, 'The nonexisting.module.bogus module \\(an optional Pyomo dependency\\) failed to import'):
        deps.bogus.hello