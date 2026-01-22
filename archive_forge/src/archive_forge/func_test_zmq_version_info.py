import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_version_info(self):
    version = zmq_version_info()
    assert version[0] in range(2, 11)