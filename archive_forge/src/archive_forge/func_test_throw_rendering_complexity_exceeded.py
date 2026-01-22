import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
def test_throw_rendering_complexity_exceeded():
    plt.rcParams['path.simplify'] = False
    xx = np.arange(2000000)
    yy = np.random.rand(2000000)
    yy[1000] = np.nan
    fig, ax = plt.subplots()
    ax.plot(xx, yy)
    with pytest.raises(OverflowError):
        fig.savefig(io.BytesIO())