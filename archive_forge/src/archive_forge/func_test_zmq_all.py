import pytest
def test_zmq_all():
    import zmq
    for name in zmq.__all__:
        assert hasattr(zmq, name)