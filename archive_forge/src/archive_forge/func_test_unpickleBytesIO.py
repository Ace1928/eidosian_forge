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
def test_unpickleBytesIO(self) -> None:
    """
        A cStringIO pickled with bytes in it will yield an L{io.BytesIO} on
        python 3.
        """
    pickledStringIWithText = b"ctwisted.persisted.styles\nunpickleStringI\np0\n(S'test'\np1\nI0\ntp2\nRp3\n."
    loaded = pickle.loads(pickledStringIWithText)
    self.assertIsInstance(loaded, io.StringIO)
    self.assertEqual(loaded.getvalue(), 'test')