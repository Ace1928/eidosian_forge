from unittest import TestCase
import zmq
from zmq.sugar import version
def test_pyzmq_version(self):
    vs = zmq.pyzmq_version()
    vs2 = zmq.__version__
    assert isinstance(vs, str)
    if zmq.__revision__:
        assert vs == '@'.join(vs2, zmq.__revision__)
    else:
        assert vs == vs2
    if version.VERSION_EXTRA:
        assert version.VERSION_EXTRA in vs
        assert version.VERSION_EXTRA in vs2