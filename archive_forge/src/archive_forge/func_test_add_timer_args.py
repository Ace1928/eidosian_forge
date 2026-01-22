import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_add_timer_args(self):

    def foo(*args, **kwargs):
        pass
    self.tg.add_timer_args(1, foo, ['arg'], {'kwarg': 'kwarg'}, initial_delay=1, stop_on_exception=False)
    self.assertEqual(1, len(self.tg.timers))
    timer = self.tg.timers[0]
    self.assertTrue(timer._running)
    self.assertEqual(('arg',), timer.args)
    self.assertEqual({'kwarg': 'kwarg'}, timer.kw)