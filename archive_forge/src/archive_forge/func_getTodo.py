import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
def getTodo(self):
    """
        Return a L{Todo} object if the test is marked todo. Checks on the
        instance first, then the class, then the module, then packages. As
        soon as it finds something with a C{todo} attribute, returns that.
        Returns L{None} if it cannot find anything. See L{TestCase} docstring
        for more details.
        """
    todo = util.acquireAttribute(self._parents, 'todo', None)
    if todo is None:
        return None
    return makeTodo(todo)