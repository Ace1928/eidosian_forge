from io import BytesIO
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
@image_comparison(['bbox_inches_tight_clipping'], remove_text=True, savefig_kwarg={'bbox_inches': 'tight'})
def test_bbox_inches_tight_clipping():
    plt.scatter(np.arange(10), np.arange(10))
    ax = plt.gca()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    patch = mpatches.Rectangle([-50, -50], 100, 100, transform=ax.transData, facecolor='blue', alpha=0.5)
    path = mpath.Path.unit_regular_star(5).deepcopy()
    path.vertices *= 0.25
    patch.set_clip_path(path, transform=ax.transAxes)
    plt.gcf().artists.append(patch)