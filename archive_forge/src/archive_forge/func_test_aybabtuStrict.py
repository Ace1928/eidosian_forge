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
def test_aybabtuStrict(self) -> None:
    """
        For a diamond-shaped inheritance graph, L{styles._aybabtu} returns a
        list containing I{both} intermediate subclasses.
        """
    self.assertEqual(styles._aybabtu(VersionedDiamondSubClass), [VersionedSubSubClass, VersionedSubClass, SecondVersionedSubClass])