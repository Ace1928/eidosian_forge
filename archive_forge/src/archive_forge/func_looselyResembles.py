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
def looselyResembles(module: ModuleType) -> Matcher[ModuleType]:
    """
    Match a module with a L{ModuleSpec} like that of the given module.

    @return: A matcher for a module spec that has the same name and origin as
        the given module spec, though the origin may be structurally inequal
        as long as it is semantically equal.
    """
    expected = module.__spec__
    assert expected is not None
    match_spec = has_properties({'name': equal_to(expected.name), 'origin': after(filepath.FilePath, equal_to(filepath.FilePath(expected.origin)))})
    return after(attrgetter('__spec__'), match_spec)