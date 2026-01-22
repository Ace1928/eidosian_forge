import sys
import pytest
from IPython.utils import capture
def test_capture_output_no_stdout():
    """test capture_output(stdout=False)"""
    rich = capture.RichOutput(data=full_data)
    with capture.capture_output(stdout=False) as cap:
        print(hello_stdout, end='')
        print(hello_stderr, end='', file=sys.stderr)
        rich.display()
    assert '' == cap.stdout
    assert hello_stderr == cap.stderr
    assert len(cap.outputs) == 1