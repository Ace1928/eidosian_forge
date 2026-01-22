import pytest
from panel import config, state
from panel.pane import HTML, panel
from panel.param import ParamFunction
from panel.tests.conftest import set_env_var
def test_config_set_console_output():
    with config.set(console_output=False):
        assert config.console_output == 'disable'
    with config.set(console_output='disable'):
        assert config.console_output == 'disable'
    with config.set(console_output='replace'):
        assert config.console_output == 'replace'
    with config.set(console_output='disable'):
        with config.set(console_output='accumulate'):
            assert config.console_output == 'accumulate'