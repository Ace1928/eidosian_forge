import pathlib
from panel.io.mime_render import (
def test_format_mime_repr_markdown():
    assert format_mime(Markdown('**BOLD**')) == ('<p><strong>BOLD</strong></p>', 'text/html')