import logging
import pytest
import panel as pn
@pytest.mark.xdist_group('debugger')
def test_debugger_constructor():
    debugger = pn.widgets.Debugger()
    assert repr(debugger).startswith('Debugger(')
    debugger = pn.widgets.Debugger(only_last=False)
    assert repr(debugger).startswith('Debugger(')
    debugger = pn.widgets.Debugger(level=20)
    assert repr(debugger).startswith('Debugger(')