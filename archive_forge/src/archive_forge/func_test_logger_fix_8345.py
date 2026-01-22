import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.logmodepython
@pytest.mark.skipif(LOG_MODE != 'PYTHON', reason='Requires KIVY_LOG_MODE==PYTHON to run.')
def test_logger_fix_8345():
    """
    The test checks that the ConsoleHandler is not in the Logger
    handlers list if stderr is None.  Test sets stderr to None,
    if the Console handler is found, the test fails.
    Pythonw and Pyinstaller 5.7+ (with console set to false) set stderr
    to None.
    """
    from kivy.logger import Logger, add_kivy_handlers, ConsoleHandler
    original_sys_stderr = sys.stderr
    sys.stderr = None
    add_kivy_handlers(Logger)
    sys.stderr = original_sys_stderr
    console_handler_found = any((isinstance(handler, ConsoleHandler) for handler in Logger.handlers))
    assert not console_handler_found, 'Console handler added, despite sys.stderr being None'