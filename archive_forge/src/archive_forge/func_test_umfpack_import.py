import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_umfpack_import():
    from skimage.segmentation import random_walker_segmentation
    UmfpackContext = random_walker_segmentation.UmfpackContext
    try:
        import scikits.umfpack
        assert UmfpackContext is not None
    except ImportError:
        assert UmfpackContext is None