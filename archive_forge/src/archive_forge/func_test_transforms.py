import copy
import matplotlib.pyplot as plt
from matplotlib.scale import (
import matplotlib.scale as mscale
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import numpy as np
from numpy.testing import assert_allclose
import io
import pytest
def test_transforms(self):
    a0 = 17.0
    a = np.linspace(-50, 50, 100)
    forward = AsinhTransform(a0)
    inverse = forward.inverted()
    invinv = inverse.inverted()
    a_forward = forward.transform_non_affine(a)
    a_inverted = inverse.transform_non_affine(a_forward)
    assert_allclose(a_inverted, a)
    a_invinv = invinv.transform_non_affine(a)
    assert_allclose(a_invinv, a0 * np.arcsinh(a / a0))