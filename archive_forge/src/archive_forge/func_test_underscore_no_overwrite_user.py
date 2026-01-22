import sys
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import CapturedIO
def test_underscore_no_overwrite_user():
    ip.run_cell('_ = 42', store_history=True)
    ip.run_cell('1+1', store_history=True)
    with AssertPrints('42'):
        ip.run_cell('print(_)', store_history=True)
    ip.run_cell('del _', store_history=True)
    ip.run_cell('6+6', store_history=True)
    with AssertPrints('12'):
        ip.run_cell('_', store_history=True)