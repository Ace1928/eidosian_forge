import numpy as np
import pytest
from skimage.io import imread, imsave, use_plugin, reset_plugins, plugin_order
from skimage._shared import testing
@pytest.fixture(autouse=True)
def use_simpleitk_plugin():
    """Ensure that SimpleITK plugin is used."""
    use_plugin('simpleitk')
    yield
    reset_plugins()