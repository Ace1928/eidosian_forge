from unittest import TestCase
import zmq
from zmq.sugar import version
def test_pyzmq_version_info(self):
    info = zmq.pyzmq_version_info()
    assert isinstance(info, tuple)
    for n in info[:3]:
        assert isinstance(n, int)
    if version.VERSION_EXTRA:
        assert len(info) == 4
        assert info[-1] == float('inf')
    else:
        assert len(info) == 3