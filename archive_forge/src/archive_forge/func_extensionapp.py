import io
import logging
import pytest
from traitlets import default
from .mockextension import MockExtensionApp
from notebook_shim import shim
@pytest.fixture
def extensionapp(jp_serverapp):
    return jp_serverapp.extension_manager.extension_points['mockextension'].app