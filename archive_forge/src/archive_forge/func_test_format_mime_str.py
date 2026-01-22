import pathlib
from panel.io.mime_render import (
def test_format_mime_str():
    assert format_mime('foo') == ('foo', 'text/plain')