import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
def test_incompatible_shapes():
    rng = np.random.default_rng(0)
    I0 = rng.normal(size=(256, 256))
    I1 = rng.normal(size=(128, 256))
    with pytest.raises(ValueError):
        u, v = optical_flow_tvl1(I0, I1)