import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def test_capture_decorator_clear_output(self):
    msg_id = 'msg-id'
    get_ipython = self._mock_get_ipython(msg_id)
    clear_output = self._mock_clear_output()
    with self._mocked_ipython(get_ipython, clear_output):
        widget = widget_output.Output()

        @widget.capture(clear_output=True, wait=True)
        def captee(*args, **kwargs):
            assert widget.msg_id == msg_id
        captee()
        captee()
    assert len(clear_output.calls) == 2
    assert clear_output.calls[0] == clear_output.calls[1] == ((), {'wait': True})