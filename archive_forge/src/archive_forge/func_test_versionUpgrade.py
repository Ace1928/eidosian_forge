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
def test_versionUpgrade(self) -> None:
    global MyVersioned

    class MyVersioned(styles.Versioned):
        persistenceVersion = 2
        persistenceForgets = ['garbagedata']
        v3 = 0
        v4 = 0

        def __init__(self) -> None:
            self.somedata = 'xxx'
            self.garbagedata = lambda q: 'cant persist'

        def upgradeToVersion3(self) -> None:
            self.v3 += 1

        def upgradeToVersion4(self) -> None:
            self.v4 += 1
    mv = MyVersioned()
    assert not (mv.v3 or mv.v4), "hasn't been upgraded yet"
    pickl = pickle.dumps(mv)
    MyVersioned.persistenceVersion = 4
    obj = pickle.loads(pickl)
    styles.doUpgrade()
    assert obj.v3, "didn't do version 3 upgrade"
    assert obj.v4, "didn't do version 4 upgrade"
    pickl = pickle.dumps(obj)
    obj = pickle.loads(pickl)
    styles.doUpgrade()
    assert obj.v3 == 1, 'upgraded unnecessarily'
    assert obj.v4 == 1, 'upgraded unnecessarily'