from __future__ import annotations
from typing import Callable, Iterable
from typing_extensions import Concatenate, ParamSpec
from twisted.python import formmethod
from twisted.trial import unittest
def testBoolean(self) -> None:
    tests = [('yes', 1), ('', 0), ('False', 0), ('no', 0)]
    self.argTest(formmethod.Boolean, tests, ())