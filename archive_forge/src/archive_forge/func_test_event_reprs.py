import pytest
import zmq
import zmq.constants
@pytest.mark.parametrize('event_name', list(zmq.Event.__members__))
def test_event_reprs(event_name):
    event = getattr(zmq.Event, event_name)
    assert event_name in repr(event)