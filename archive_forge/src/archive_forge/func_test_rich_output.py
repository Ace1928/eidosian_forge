import sys
import pytest
from IPython.utils import capture
def test_rich_output():
    """test RichOutput basics"""
    data = basic_data
    metadata = basic_metadata
    rich = capture.RichOutput(data=data, metadata=metadata)
    assert rich._repr_html_() == data['text/html']
    assert rich._repr_png_() == (data['image/png'], metadata['image/png'])
    assert rich._repr_latex_() is None
    assert rich._repr_javascript_() is None
    assert rich._repr_svg_() is None