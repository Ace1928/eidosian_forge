from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_patchAlreadyPatched(self) -> None:
    """
        Adding a patch for an object and attribute that already have a patch
        overrides the existing patch.
        """
    self.monkeyPatcher.addPatch(self.testObject, 'foo', 'blah')
    self.monkeyPatcher.addPatch(self.testObject, 'foo', 'BLAH')
    self.monkeyPatcher.patch()
    self.assertEqual(self.testObject.foo, 'BLAH')
    self.monkeyPatcher.restore()
    self.assertEqual(self.testObject.foo, self.originalObject.foo)