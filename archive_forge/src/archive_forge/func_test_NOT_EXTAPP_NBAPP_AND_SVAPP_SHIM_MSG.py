import io
import logging
import pytest
from traitlets import default
from .mockextension import MockExtensionApp
from notebook_shim import shim
@pytest.mark.parametrize('jp_argv,trait_name,trait_value', list_test_params([('jinja_environment_options', {}), ('jinja_template_vars', {}), ('extra_template_paths', []), ('quit_button', True)]))
def test_NOT_EXTAPP_NBAPP_AND_SVAPP_SHIM_MSG(read_app_logs, extensionapp, jp_argv, trait_name, trait_value):
    log = read_app_logs()
    log_msg = shim.NOT_EXTAPP_NBAPP_AND_SVAPP_SHIM_MSG(trait_name, 'MockExtensionApp')
    assert log_msg in log
    assert getattr(extensionapp.serverapp, trait_name) == trait_value