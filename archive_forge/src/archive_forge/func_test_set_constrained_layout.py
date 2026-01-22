import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@pytest.mark.parametrize('arg, state', [(True, True), (False, False), ({}, True), ({'rect': None}, True)])
def test_set_constrained_layout(arg, state):
    fig, ax = plt.subplots(constrained_layout=arg)
    assert fig.get_constrained_layout() is state