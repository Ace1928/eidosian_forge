import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
@image_comparison(['fft_peaks'], remove_text=True)
def test_fft_peaks():
    fig, ax = plt.subplots()
    t = np.arange(65536)
    p1 = ax.plot(abs(np.fft.fft(np.sin(2 * np.pi * 0.01 * t) * np.blackman(len(t)))))
    fig.canvas.draw()
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = path.cleaned(simplify=True)
    assert simplified.vertices.size == 36