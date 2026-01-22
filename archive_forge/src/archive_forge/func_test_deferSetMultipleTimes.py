from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_deferSetMultipleTimes(self) -> None:
    """
        L{crefutil._Defer} can be assigned a key only one time.
        """
    d = crefutil._Defer()
    d[0] = 1
    self.assertRaises(RuntimeError, d.__setitem__, 0, 1)