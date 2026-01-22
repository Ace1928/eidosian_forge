import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
@pytest.mark.parametrize('shape, interpolation, output_size', itertools.product(shapes_radon_iradon_circle, interpolations, output_sizes))
def test_radon_iradon_circle(shape, interpolation, output_size):
    check_radon_iradon_circle(interpolation, shape, output_size)