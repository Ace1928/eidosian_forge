import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_notify_register_deregister(self):

    def call_me(state, details):
        pass

    class A(object):

        def call_me_too(self, state, details):
            pass
    notifier = nt.Notifier()
    notifier.register(nt.Notifier.ANY, call_me)
    a = A()
    notifier.register(nt.Notifier.ANY, a.call_me_too)
    self.assertEqual(2, len(notifier))
    notifier.deregister(nt.Notifier.ANY, call_me)
    notifier.deregister(nt.Notifier.ANY, a.call_me_too)
    self.assertEqual(0, len(notifier))