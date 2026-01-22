from __future__ import annotations
from typing import Callable, Iterable
from typing_extensions import Concatenate, ParamSpec
from twisted.python import formmethod
from twisted.trial import unittest
def test_argument(self) -> None:
    """
        Test that corce correctly raises NotImplementedError.
        """
    arg = formmethod.Argument('name')
    self.assertRaises(NotImplementedError, arg.coerce, '')