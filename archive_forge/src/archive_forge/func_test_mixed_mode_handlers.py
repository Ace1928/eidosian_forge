import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.logmodemixed
@pytest.mark.skipif(LOG_MODE != 'MIXED', reason='Requires KIVY_LOG_MODE==MIXED to run.')
def test_mixed_mode_handlers():
    assert not are_regular_logs_handled()
    assert are_kivy_logger_logs_handled()
    assert not is_stderr_output_handled()