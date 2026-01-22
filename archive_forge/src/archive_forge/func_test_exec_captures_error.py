import pathlib
from panel.io.mime_render import (
def test_exec_captures_error():

    def capture_stderr(stderr):
        print()
    stderr = WriteCallbackStream(capture_stderr)
    assert exec_with_return('raise ValueError("bar")', stderr=stderr) is None
    assert 'ValueError: bar' in stderr.getvalue()