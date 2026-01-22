import numpy as np
import pytest
from skimage.io import imread, imsave, use_plugin, reset_plugins, plugin_order
from skimage._shared import testing
Ensure that SimpleITK plugin is used.