import os
import sys
import pytest
from nipype import config
from unittest.mock import MagicMock
def test_display_noconfig_nosystem_notinstalled(monkeypatch):
    """
    Check that an exception is raised if xvfbwrapper is not installed
    but necessary (no config and $DISPLAY unset)
    """
    config._display = None
    if config.has_option('execution', 'display_variable'):
        config._config.remove_option('execution', 'display_variable')
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setitem(sys.modules, 'xvfbwrapper', None)
    with pytest.raises(RuntimeError):
        config.get_display()