import logging
import pytest
import panel as pn
@pytest.mark.xdist_group('debugger')
def test_debugger_logging_info():
    logger = logging.getLogger('panel.callbacks')
    debugger = pn.widgets.Debugger(level=logging.DEBUG)
    msg = 'debugger test message'
    logger.info(msg)
    assert msg in debugger.terminal.output
    assert debugger.title == 'i: 1'
    msg = 'debugger test warning'
    logger.warning(msg)
    assert msg in debugger.terminal.output
    assert 'w: </span>1' in debugger.title