import threading
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase
def build_device(self, mon_sub=b'', in_prefix=b'in', out_prefix=b'out'):
    self.device = devices.ThreadMonitoredQueue(zmq.PAIR, zmq.PAIR, zmq.PUB, in_prefix, out_prefix)
    alice = self.context.socket(zmq.PAIR)
    bob = self.context.socket(zmq.PAIR)
    mon = self.context.socket(zmq.SUB)
    aport = alice.bind_to_random_port('tcp://127.0.0.1')
    bport = bob.bind_to_random_port('tcp://127.0.0.1')
    mport = mon.bind_to_random_port('tcp://127.0.0.1')
    mon.setsockopt(zmq.SUBSCRIBE, mon_sub)
    self.device.connect_in('tcp://127.0.0.1:%i' % aport)
    self.device.connect_out('tcp://127.0.0.1:%i' % bport)
    self.device.connect_mon('tcp://127.0.0.1:%i' % mport)
    self.device.start()
    time.sleep(0.2)
    try:
        mon.recv_multipart(zmq.NOBLOCK)
    except zmq.ZMQError:
        pass
    self.sockets.extend([alice, bob, mon])
    return (alice, bob, mon)