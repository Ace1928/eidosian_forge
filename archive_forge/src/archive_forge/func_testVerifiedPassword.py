from __future__ import annotations
from typing import Callable, Iterable
from typing_extensions import Concatenate, ParamSpec
from twisted.python import formmethod
from twisted.trial import unittest
def testVerifiedPassword(self) -> None:
    goodTests = {('foo', 'foo'): 'foo', ('ab', 'ab'): 'ab'}.items()
    badTests = [('ab', 'a'), ('12345', '12345'), ('', ''), ('a', 'a'), ('a',), ('a', 'a', 'a')]
    self.argTest(formmethod.VerifiedPassword, goodTests, badTests, min=2, max=4)