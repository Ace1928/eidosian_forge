import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def test_capture_decorator(self):
    msg_id = 'msg-id'
    get_ipython = self._mock_get_ipython(msg_id)
    clear_output = self._mock_clear_output()
    expected_argument = 'arg'
    expected_keyword_argument = True
    captee_calls = []
    with self._mocked_ipython(get_ipython, clear_output):
        widget = widget_output.Output()
        assert widget.msg_id == ''

        @widget.capture()
        def captee(*args, **kwargs):
            assert widget.msg_id == msg_id
            captee_calls.append((args, kwargs))
        captee(expected_argument, keyword_argument=expected_keyword_argument)
        assert widget.msg_id == ''
        captee()
    assert len(captee_calls) == 2
    assert captee_calls[0] == ((expected_argument,), {'keyword_argument': expected_keyword_argument})
    assert captee_calls[1] == ((), {})