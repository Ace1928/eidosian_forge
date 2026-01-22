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
def test_funkyReferences(self) -> None:
    o = EvilSourceror(EvilSourceror([]))
    j1 = aot.jellyToAOT(o)
    oj = aot.unjellyFromAOT(j1)
    assert oj.a is oj
    assert oj.a.b is oj.b
    assert oj.c is not oj.c.c