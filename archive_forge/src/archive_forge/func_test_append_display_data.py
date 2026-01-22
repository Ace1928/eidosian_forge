import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def test_append_display_data():
    widget = widget_output.Output()
    widget.append_display_data(Markdown('# snakes!'))
    expected = ({'output_type': 'display_data', 'data': {'text/plain': '<IPython.core.display.Markdown object>', 'text/markdown': '# snakes!'}, 'metadata': {}},)
    assert widget.outputs == expected, repr(widget.outputs)
    image_data = b'foobar'
    widget.append_display_data(Image(image_data, width=123, height=456))
    expected1 = expected + ({'output_type': 'display_data', 'data': {'image/png': 'Zm9vYmFy\n', 'text/plain': '<IPython.core.display.Image object>'}, 'metadata': {'image/png': {'width': 123, 'height': 456}}},)
    expected2 = expected + ({'output_type': 'display_data', 'data': {'image/png': 'Zm9vYmFy', 'text/plain': '<IPython.core.display.Image object>'}, 'metadata': {'image/png': {'width': 123, 'height': 456}}},)
    assert widget.outputs == expected1 or widget.outputs == expected2