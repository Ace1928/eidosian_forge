from contextlib import ExitStack
import itertools
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
import matplotlib.patches as mpatch
from matplotlib.projections import register_projection
@image_comparison(['skew_axes'], remove_text=True)
def test_set_line_coll_dash_image():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='skewx')
    ax.set_xlim(-50, 50)
    ax.set_ylim(50, -50)
    ax.grid(True)
    ax.axvline(0, color='b')