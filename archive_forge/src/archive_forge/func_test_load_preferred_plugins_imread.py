from contextlib import contextmanager
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage import io
from skimage.io import manage_plugins
@pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
def test_load_preferred_plugins_imread():
    from skimage.io._plugins import pil_plugin, matplotlib_plugin
    with protect_preferred_plugins():
        manage_plugins.preferred_plugins['imread'] = ['pil']
        manage_plugins.reset_plugins()
        plug, func = manage_plugins.plugin_store['imread'][0]
        assert func == pil_plugin.imread
        plug, func = manage_plugins.plugin_store['imshow'][0]
        assert func == matplotlib_plugin.imshow, func.__module__