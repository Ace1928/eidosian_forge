from io import BytesIO
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
def test_tight_pcolorfast():
    fig, ax = plt.subplots()
    ax.pcolorfast(np.arange(4).reshape((2, 2)))
    ax.set(ylim=(0, 0.1))
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    height, width, _ = plt.imread(buf).shape
    assert width > height