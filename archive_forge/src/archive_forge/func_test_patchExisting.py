from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_patchExisting(self) -> None:
    """
        Patching an attribute that exists sets it to the value defined in the
        patch.
        """
    self.monkeyPatcher.addPatch(self.testObject, 'foo', 'haha')
    self.monkeyPatcher.patch()
    self.assertEqual(self.testObject.foo, 'haha')