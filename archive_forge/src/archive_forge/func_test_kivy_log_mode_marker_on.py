import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.logmodepython
@pytest.mark.logmodemixed
@pytest.mark.skipif(LOG_MODE == 'KIVY', reason='Requires KIVY_LOG_MODE!=KIVY to run.')
def test_kivy_log_mode_marker_on():
    """
    This is a test of the pytest markers.
    This should only be invoked if the environment variable is appropriately set
    (before pytest is run).

    Also, tests that kivy.logger paid attention to the environment variable
    """
    assert logging.root.level != 0, 'Root logger was modified'