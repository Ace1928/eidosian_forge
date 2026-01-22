from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_run_with_patches_restores_on_exception(self):

    def _():
        self.assertEqual(self.test_object.foo, 'haha')
        self.assertEqual(self.test_object.bar, 'blahblah')
        raise RuntimeError('Something went wrong!')
    self.monkey_patcher.add_patch(self.test_object, 'foo', 'haha')
    self.monkey_patcher.add_patch(self.test_object, 'bar', 'blahblah')
    self.assertThat(lambda: self.monkey_patcher.run_with_patches(_), Raises(MatchesException(RuntimeError('Something went wrong!'))))
    self.assertEqual(self.test_object.foo, self.original_object.foo)
    self.assertEqual(self.test_object.bar, self.original_object.bar)