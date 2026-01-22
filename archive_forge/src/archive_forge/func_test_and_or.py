import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_and_or(self):
    mod0, avail0 = attempt_import('ply', defer_check=True)
    mod1, avail1 = attempt_import('pyomo.common.tests.dep_mod', defer_check=True)
    mod2, avail2 = attempt_import('pyomo.common.tests.dep_mod', minimum_version='2.0', defer_check=True)
    _and = avail0 & avail1
    self.assertIsInstance(_and, _DeferredAnd)
    _or = avail1 | avail2
    self.assertIsInstance(_or, _DeferredOr)
    self.assertIsNone(avail0._available)
    self.assertIsNone(avail1._available)
    self.assertIsNone(avail2._available)
    self.assertTrue(_or)
    self.assertIsNone(avail0._available)
    self.assertTrue(avail1._available)
    self.assertIsNone(avail2._available)
    self.assertTrue(_and)
    self.assertTrue(avail0._available)
    self.assertTrue(avail1._available)
    self.assertIsNone(avail2._available)
    _and_and = avail0 & avail1 & avail2
    self.assertFalse(_and_and)
    _and_or = avail0 & avail1 | avail2
    self.assertTrue(_and_or)
    _or_and = avail0 | avail2 & avail2
    self.assertTrue(_or_and)
    _or_and = (avail0 | avail2) & avail2
    self.assertFalse(_or_and)
    _or_or = avail0 | avail1 | avail2
    self.assertTrue(_or_or)
    _rand = True & avail1
    self.assertIsInstance(_rand, _DeferredAnd)
    self.assertTrue(_rand)
    _ror = False | avail1
    self.assertIsInstance(_ror, _DeferredOr)
    self.assertTrue(_ror)