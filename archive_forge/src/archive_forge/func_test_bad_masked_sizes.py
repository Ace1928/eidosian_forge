import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_bad_masked_sizes():
    """Test error handling when given differing sized masked arrays."""
    x = np.arange(3)
    y = np.arange(3)
    u = np.ma.array(15.0 * np.ones((4,)))
    v = np.ma.array(15.0 * np.ones_like(u))
    u[1] = np.ma.masked
    v[1] = np.ma.masked
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.barbs(x, y, u, v)