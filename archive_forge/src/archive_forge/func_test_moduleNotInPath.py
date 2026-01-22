from __future__ import annotations
import os
import sys
import unittest as pyunit
from hashlib import md5
from operator import attrgetter
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator
from hamcrest import assert_that, equal_to, has_properties
from hamcrest.core.matcher import Matcher
from twisted.python import filepath, util
from twisted.python.modules import PythonAttribute, PythonModule, getModule
from twisted.python.reflect import ModuleNotFound
from twisted.trial import reporter, runner, unittest
from twisted.trial._asyncrunner import _iterateTests
from twisted.trial.itrial import ITestCase
from twisted.trial.test import packages
from .matchers import after
def test_moduleNotInPath(self) -> None:
    """
        If passed the path to a file containing the implementation of a
        module within a package which is not on the import path,
        L{runner.filenameToModule} returns a module object loosely
        resembling the module defined by that file anyway.
        """
    self.mangleSysPath(self.oldPath)
    sample1 = runner.filenameToModule(os.path.join(self.parent, 'goodpackage', 'test_sample.py'))
    self.assertEqual(sample1.__name__, 'goodpackage.test_sample')
    self.cleanUpModules()
    self.mangleSysPath(self.newPath)
    from goodpackage import test_sample as sample2
    self.assertIsNot(sample1, sample2)
    assert_that(sample1, looselyResembles(sample2))