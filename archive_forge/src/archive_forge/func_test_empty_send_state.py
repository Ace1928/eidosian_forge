from traitlets import Bool, Tuple, List
from .utils import setup, teardown, DummyComm
from ..widget import Widget
from ..._version import __control_protocol_version__
def test_empty_send_state():
    w = SimpleWidget()
    w.send_state([])
    assert w.comm.messages == []