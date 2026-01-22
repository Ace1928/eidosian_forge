import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def test_append_stdout():
    widget = widget_output.Output()
    widget.append_stdout('snakes!')
    expected = (_make_stream_output('snakes!', 'stdout'),)
    assert widget.outputs == expected, repr(widget.outputs)
    widget.append_stdout('more snakes!')
    expected += (_make_stream_output('more snakes!', 'stdout'),)
    assert widget.outputs == expected, repr(widget.outputs)