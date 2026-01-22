import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_not_callable(self):
    notifier = nt.Notifier()
    self.assertRaises(ValueError, notifier.register, nt.Notifier.ANY, 2)