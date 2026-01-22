import os
import sys
import pytest
from nipype import config
from unittest.mock import MagicMock
@pytest.mark.parametrize('dispnum', range(5))
def test_display_system(monkeypatch, dispnum):
    """Check that when only a $DISPLAY is defined, it is used"""
    config._display = None
    config._config.remove_option('execution', 'display_variable')
    dispstr = ':%d' % dispnum
    monkeypatch.setenv('DISPLAY', dispstr)
    assert config.get_display() == dispstr
    assert config.get_display() == dispstr