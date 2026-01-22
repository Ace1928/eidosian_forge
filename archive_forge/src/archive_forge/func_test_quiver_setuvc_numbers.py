import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_quiver_setuvc_numbers():
    """Check that it is possible to set all arrow UVC to the same numbers"""
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(2), np.arange(2))
    U = V = np.ones_like(X)
    q = ax.quiver(X, Y, U, V)
    q.set_UVC(0, 1)