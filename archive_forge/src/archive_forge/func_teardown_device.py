import threading
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase
def teardown_device(self):
    for i in range(50):
        context = getattr(self.device, '_context', None)
        if context is not None:
            break
        time.sleep(0.1)
    if context is not None:
        t = threading.Thread(target=self.device._context.term, daemon=True)
        t.start()
    for socket in self.sockets:
        socket.close()
    if context is not None:
        t.join(timeout=5)
    self.device.join(timeout=5)