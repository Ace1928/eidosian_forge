import numpy as np
import matplotlib.pyplot as plt
import matplotlib.projections as mprojections
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist.axislines import Subplot
from mpl_toolkits.axisartist.floating_axes import (
from mpl_toolkits.axisartist.grid_finder import FixedLocator
from mpl_toolkits.axisartist import angle_helper
def test_subplot():
    fig = plt.figure(figsize=(5, 5))
    ax = Subplot(fig, 111)
    fig.add_subplot(ax)