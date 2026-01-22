import sys
import pytest
from IPython.utils import capture
def test_rich_output_display():
    """test RichOutput.display

    This is a bit circular, because we are actually using the capture code we are testing
    to test itself.
    """
    data = full_data
    rich = capture.RichOutput(data=data)
    with capture.capture_output() as cap:
        rich.display()
    assert len(cap.outputs) == 1
    rich2 = cap.outputs[0]
    assert rich2.data == rich.data
    assert rich2.metadata == rich.metadata