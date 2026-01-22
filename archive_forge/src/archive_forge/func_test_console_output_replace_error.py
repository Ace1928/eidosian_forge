import pytest
from panel import config, state
from panel.pane import HTML, panel
from panel.param import ParamFunction
from panel.tests.conftest import set_env_var
def test_console_output_replace_error(document, comm, get_display_handle):
    pane = HTML()
    with set_env_var('PANEL_CONSOLE_OUTPUT', 'replace'):
        model = pane.get_root(document, comm)
        handle = get_display_handle(model)
        try:
            1 / 0
        except Exception as e:
            pane._on_error(model.ref['id'], e)
        assert 'text/html' in handle
        assert 'ZeroDivisionError' in handle['text/html']
        try:
            1 + '2'
        except Exception as e:
            pane._on_error(model.ref['id'], e)
        assert 'text/html' in handle
        assert 'ZeroDivisionError' not in handle['text/html']
        assert 'TypeError' in handle['text/html']
        pane._cleanup(model)
        assert model.ref['id'] not in state._handles