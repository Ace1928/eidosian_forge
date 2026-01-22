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
def test_sortCases(self) -> None:
    from twisted.trial.test import sample
    suite = self.loader.loadClass(sample.AlphabetTest)
    self.assertEqual(['test_a', 'test_b', 'test_c'], [test._testMethodName for test in suite._tests])
    newOrder = ['test_b', 'test_c', 'test_a']
    sortDict = dict(zip(newOrder, range(3)))
    self.loader.sorter = lambda x: sortDict.get(x.shortDescription(), -1)
    suite = self.loader.loadClass(sample.AlphabetTest)
    self.assertEqual(newOrder, [test._testMethodName for test in suite._tests])