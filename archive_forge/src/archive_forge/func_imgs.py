import os
import numpy as np
from skimage.io import use_plugin, reset_plugins
from skimage.io.collection import MultiImage
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose
from pytest import fixture
@fixture
def imgs():
    use_plugin('pil')
    paths = [testing.fetch('data/multipage_rgb.tif'), testing.fetch('data/no_time_for_that_tiny.gif')]
    imgs = [MultiImage(paths[0]), MultiImage(paths[0], conserve_memory=False), MultiImage(paths[1]), MultiImage(paths[1], conserve_memory=False), MultiImage(os.pathsep.join(paths))]
    yield imgs
    reset_plugins()