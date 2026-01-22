import pytest
from panel import config, state
from panel.pane import HTML, panel
from panel.param import ParamFunction
from panel.tests.conftest import set_env_var
def test_console_output_replace_stdout(document, comm, get_display_handle):
    pane = HTML()
    with set_env_var('PANEL_CONSOLE_OUTPUT', 'replace'):
        model = pane.get_root(document, comm)
        handle = get_display_handle(model)
        pane._on_stdout(model.ref['id'], ['print output'])
        assert handle == {'text/html': 'print output</br>', 'raw': True}
        pane._on_stdout(model.ref['id'], ['new output'])
        assert handle == {'text/html': 'new output</br>', 'raw': True}
        pane._cleanup(model)
        assert model.ref['id'] not in state._handles