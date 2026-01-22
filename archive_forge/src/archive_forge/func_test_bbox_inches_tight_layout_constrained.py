from io import BytesIO
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
@image_comparison(['bbox_inches_tight_layout.png'], remove_text=True, style='mpl20', savefig_kwarg=dict(bbox_inches='tight', pad_inches='layout'))
def test_bbox_inches_tight_layout_constrained():
    fig, ax = plt.subplots(layout='constrained')
    fig.get_layout_engine().set(h_pad=0.5)
    ax.set_aspect('equal')