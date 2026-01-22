import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_cancel_timeout(self):

    def foo(*args, **kwargs):
        time.sleep(0.3)
    self.tg.add_thread(foo, 'arg', kwarg='kwarg')
    time.sleep(0)
    self.tg.cancel(timeout=0.2, wait_time=0.1)
    self.assertEqual(0, len(self.tg.threads))