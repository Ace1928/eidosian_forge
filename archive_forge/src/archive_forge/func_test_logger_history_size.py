import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.xfail
def test_logger_history_size():
    from kivy.logger import Logger, LoggerHistory
    for x in range(200):
        Logger.info(x)
    assert len(LoggerHistory.history) == 100, 'Wrong size: %s' % len(LoggerHistory.history)