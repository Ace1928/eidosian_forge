import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_layout_leak():
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    fig.add_subplot()
    fig.draw_without_rendering()
    plt.close('all')
    del fig
    gc.collect()
    assert not any((isinstance(obj, mpl._layoutgrid.LayoutGrid) for obj in gc.get_objects()))