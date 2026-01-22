from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_patch_returns_cleanup(self):
    test_object = TestObj()
    original = test_object.foo
    cleanup = patch(test_object, 'foo', 42)
    cleanup()
    self.assertEqual(original, test_object.foo)