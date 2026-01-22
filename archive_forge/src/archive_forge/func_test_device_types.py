import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest, have_gevent
def test_device_types(self):
    for devtype in (zmq.STREAMER, zmq.FORWARDER, zmq.QUEUE):
        dev = devices.Device(devtype, zmq.PAIR, zmq.PAIR)
        assert dev.device_type == devtype
        del dev