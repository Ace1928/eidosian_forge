import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_data_truncate(echo):
    w = TruncateDataWidget()
    data = memoryview(b'x' * 30)
    w.set_state(dict(a=True, d={'data': data}))
    assert len(w.comm.messages) == 2 if echo else 1
    msg = w.comm.messages[-1]
    buffers = msg[1].pop('buffers')
    assert msg == ((), dict(data=dict(method='update', state=dict(d={}), buffer_paths=[['d', 'data']])))
    assert len(buffers) == 1
    assert buffers[0] == data[:20].tobytes()