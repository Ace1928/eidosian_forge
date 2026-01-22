import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_restricted_notifier_no_any(self):
    notifier = nt.RestrictedNotifier(['a', 'b'], allow_any=False)
    self.assertRaises(ValueError, notifier.register, nt.RestrictedNotifier.ANY, lambda *args, **kargs: None)
    notifier.register('b', lambda *args, **kargs: None)
    self.assertEqual(1, len(notifier))