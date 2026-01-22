import pytest
from panel import config, state
from panel.pane import HTML, panel
from panel.param import ParamFunction
from panel.tests.conftest import set_env_var
def test_env_var_console_output():
    with set_env_var('PANEL_CONSOLE_OUTPUT', 'disable'):
        assert config.console_output == 'disable'
    with set_env_var('PANEL_CONSOLE_OUTPUT', 'replace'):
        assert config.console_output == 'replace'
    with config.set(console_output='disable'):
        with set_env_var('PANEL_DOC_BUILD', 'accumulate'):
            assert config.console_output == 'disable'