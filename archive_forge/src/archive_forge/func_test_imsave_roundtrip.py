from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
from skimage.io import imread, imsave, use_plugin, reset_plugins
from skimage._shared import testing
from skimage._shared.testing import (
from pytest import importorskip
importorskip('imread')
def test_imsave_roundtrip(self):
    dtype = np.uint8
    np.random.seed(0)
    for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
        x = np.ones(shape, dtype=dtype) * np.random.rand(*shape)
        if np.issubdtype(dtype, np.floating):
            yield (self.roundtrip, x, 255)
        else:
            x = (x * 255).astype(dtype)
            yield (self.roundtrip, x)