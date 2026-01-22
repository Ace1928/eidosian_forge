import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@pytest.mark.parametrize('label', ['xlabel', 'ylabel'])
def test_verybig_decorators(label):
    """Test that no warning emitted when xlabel/ylabel too big."""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set(**{label: 'a' * 100})