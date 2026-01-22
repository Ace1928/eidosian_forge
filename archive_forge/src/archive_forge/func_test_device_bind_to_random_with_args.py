import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest, have_gevent
def test_device_bind_to_random_with_args(self):
    dev = devices.ThreadDevice(zmq.PULL, zmq.PUSH, -1)
    iface = 'tcp://127.0.0.1'
    ports = []
    min, max = (5000, 5050)
    ports.extend([dev.bind_in_to_random_port(iface, min_port=min, max_port=max), dev.bind_out_to_random_port(iface, min_port=min, max_port=max)])
    for port in ports:
        if port < min or port > max:
            self.fail('Unexpected port number: %i' % port)