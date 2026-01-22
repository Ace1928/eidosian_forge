import pytest
def test_eventloop():
    """test eventloop imports"""
    pytest.importorskip('tornado')
    import zmq.eventloop
    from zmq.eventloop import ioloop, zmqstream