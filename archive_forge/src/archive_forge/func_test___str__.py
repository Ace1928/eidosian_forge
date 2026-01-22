from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
def test___str__(self):
    matcher = ReturnsUnlockable(StubTree(True))
    self.assertEqual('ReturnsUnlockable(lockable_thing=I am da tree)', str(matcher))