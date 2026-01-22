import pathlib
from panel.io.mime_render import (
def test_format_mime_repr_javascript():
    assert format_mime(Javascript('1+1')) == ('<script>1+1</script>', 'text/html')