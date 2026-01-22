import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def test_append_stderr():
    widget = widget_output.Output()
    widget.append_stderr('snakes!')
    expected = (_make_stream_output('snakes!', 'stderr'),)
    assert widget.outputs == expected, repr(widget.outputs)
    widget.append_stderr('more snakes!')
    expected += (_make_stream_output('more snakes!', 'stderr'),)
    assert widget.outputs == expected, repr(widget.outputs)