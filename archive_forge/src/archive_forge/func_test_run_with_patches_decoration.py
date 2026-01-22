from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_run_with_patches_decoration(self):
    log = []

    def f(a, b, c=None):
        log.append((a, b, c))
        return 'foo'
    result = self.monkey_patcher.run_with_patches(f, 1, 2, c=10)
    self.assertEqual('foo', result)
    self.assertEqual([(1, 2, 10)], log)