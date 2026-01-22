import pathlib
from panel.io.mime_render import (
def capture_stdout(stdout):
    assert stdout == 'foo'