import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
def test_3d_motion():
    rng = np.random.default_rng(0)
    image0 = rng.normal(size=(100, 100, 100))
    gt_flow, image1 = _sin_flow_gen(image0)
    flow = optical_flow_tvl1(image0, image1, attachment=10)
    assert abs(flow - gt_flow).mean() < 0.5