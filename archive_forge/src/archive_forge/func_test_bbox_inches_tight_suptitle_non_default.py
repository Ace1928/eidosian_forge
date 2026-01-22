from io import BytesIO
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
@image_comparison(['bbox_inches_tight_suptile_non_default.png'], savefig_kwarg={'bbox_inches': 'tight'}, tol=0.1)
def test_bbox_inches_tight_suptitle_non_default():
    fig, ax = plt.subplots()
    fig.suptitle('Booo', x=0.5, y=1.1)