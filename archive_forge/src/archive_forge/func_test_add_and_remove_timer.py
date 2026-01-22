import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_add_and_remove_timer(self):

    def foo(*args, **kwargs):
        pass
    timer = self.tg.add_timer('1234', foo)
    self.assertEqual(1, len(self.tg.timers))
    timer.stop()
    self.assertEqual(1, len(self.tg.timers))
    self.tg.timer_done(timer)
    self.assertEqual(0, len(self.tg.timers))