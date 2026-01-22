import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_notify_not_called(self):
    call_collector = []

    def call_me(state, details):
        call_collector.append((state, details))
    notifier = nt.Notifier()
    notifier.register(nt.Notifier.ANY, call_me)
    notifier.notify(nt.Notifier.ANY, {})
    self.assertFalse(notifier.can_trigger_notification(nt.Notifier.ANY))
    self.assertEqual(0, len(call_collector))
    self.assertEqual(1, len(notifier))