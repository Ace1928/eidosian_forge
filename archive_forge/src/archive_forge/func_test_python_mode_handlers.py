import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.logmodepython
@pytest.mark.skipif(LOG_MODE != 'PYTHON', reason='Requires KIVY_LOG_MODE==PYTHON to run.')
def test_python_mode_handlers():
    assert not are_regular_logs_handled()
    assert not are_kivy_logger_logs_handled()
    assert not is_stderr_output_handled()