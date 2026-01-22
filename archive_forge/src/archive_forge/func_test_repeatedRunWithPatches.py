from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_repeatedRunWithPatches(self) -> None:
    """
        We should be able to call the same function with runWithPatches more
        than once. All patches should apply for each call.
        """

    def f() -> tuple[str, str, str]:
        return (self.testObject.foo, self.testObject.bar, self.testObject.baz)
    self.monkeyPatcher.addPatch(self.testObject, 'foo', 'haha')
    result = self.monkeyPatcher.runWithPatches(f)
    self.assertEqual(('haha', self.originalObject.bar, self.originalObject.baz), result)
    result = self.monkeyPatcher.runWithPatches(f)
    self.assertEqual(('haha', self.originalObject.bar, self.originalObject.baz), result)