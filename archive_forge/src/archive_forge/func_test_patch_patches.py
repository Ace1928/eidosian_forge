from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_patch_patches(self):
    test_object = TestObj()
    patch(test_object, 'foo', 42)
    self.assertEqual(42, test_object.foo)