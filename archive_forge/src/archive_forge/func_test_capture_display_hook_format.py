import sys
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import CapturedIO
def test_capture_display_hook_format():
    """Tests that the capture display hook conforms to the CapturedIO output format"""
    hook = CapturingDisplayHook(ip)
    hook({'foo': 'bar'})
    captured = CapturedIO(sys.stdout, sys.stderr, hook.outputs)
    captured.outputs