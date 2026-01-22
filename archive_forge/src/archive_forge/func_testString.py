from __future__ import annotations
from typing import Callable, Iterable
from typing_extensions import Concatenate, ParamSpec
from twisted.python import formmethod
from twisted.trial import unittest
def testString(self) -> None:
    self.argTest(formmethod.String, [('a', 'a'), (1, '1'), ('', '')], ())
    self.argTest(formmethod.String, [('ab', 'ab'), ('abc', 'abc')], ('2', ''), min=2)
    self.argTest(formmethod.String, [('ab', 'ab'), ('a', 'a')], ('223213', '345x'), max=3)
    self.argTest(formmethod.String, [('ab', 'ab'), ('add', 'add')], ('223213', 'x'), min=2, max=3)