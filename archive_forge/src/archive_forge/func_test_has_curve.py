from pytest import mark
import zmq
@only_bundled
def test_has_curve():
    """bundled libzmq has curve support"""
    assert zmq.has('curve')