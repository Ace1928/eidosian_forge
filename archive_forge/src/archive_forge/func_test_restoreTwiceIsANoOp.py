from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_restoreTwiceIsANoOp(self) -> None:
    """
        Restoring an already-restored monkey patch is a no-op.
        """
    self.monkeyPatcher.addPatch(self.testObject, 'foo', 'blah')
    self.monkeyPatcher.patch()
    self.monkeyPatcher.restore()
    self.assertEqual(self.testObject.foo, self.originalObject.foo)
    self.monkeyPatcher.restore()
    self.assertEqual(self.testObject.foo, self.originalObject.foo)