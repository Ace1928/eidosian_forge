import logging
import time
import zmq
from zmq.log import handlers
from zmq.tests import BaseZMQTestCase
def test_init_socket(self):
    pub, sub = self.create_bound_pair(zmq.PUB, zmq.SUB)
    logger = self.logger
    handler = handlers.PUBHandler(pub)
    handler.setLevel(logging.DEBUG)
    handler.root_topic = self.topic
    logger.addHandler(handler)
    assert handler.socket is pub
    assert handler.ctx is pub.context
    assert handler.ctx is self.context
    sub.setsockopt(zmq.SUBSCRIBE, self.topic.encode())
    import time
    time.sleep(0.1)
    msg1 = 'message'
    logger.info(msg1)
    topic, msg2 = sub.recv_multipart()
    assert topic == b'zmq.INFO'
    assert msg2 == (msg1 + '\n').encode('utf8')
    logger.removeHandler(handler)