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
def test_upgradeDeserializesObjectsRequiringUpgrade(self) -> None:
    global ToyClassA, ToyClassB

    class ToyClassA(styles.Versioned):
        pass

    class ToyClassB(styles.Versioned):
        pass
    x = ToyClassA()
    y = ToyClassB()
    pklA, pklB = (pickle.dumps(x), pickle.dumps(y))
    del x, y
    ToyClassA.persistenceVersion = 1

    def upgradeToVersion1(self: Any) -> None:
        self.y = pickle.loads(pklB)
        styles.doUpgrade()
    ToyClassA.upgradeToVersion1 = upgradeToVersion1
    ToyClassB.persistenceVersion = 1

    def setUpgraded(self: object) -> None:
        setattr(self, 'upgraded', True)
    ToyClassB.upgradeToVersion1 = setUpgraded
    x = pickle.loads(pklA)
    styles.doUpgrade()
    self.assertTrue(x.y.upgraded)