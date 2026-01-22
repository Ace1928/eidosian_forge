import numpy as np
import matplotlib as mpl
from matplotlib.colors import same_color
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
def test_handlerline3d():
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter([0, 1], [0, 1], marker='v')
    handles = [art3d.Line3D([0], [0], [0], marker='v')]
    leg = ax.legend(handles, ['Aardvark'], numpoints=1)
    assert handles[0].get_marker() == leg.legend_handles[0].get_marker()