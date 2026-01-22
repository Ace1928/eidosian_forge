import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['test_bboxtight.png'], remove_text=True, style='mpl20', savefig_kwarg={'bbox_inches': 'tight'})
def test_bboxtight():
    fig, ax = plt.subplots(layout='constrained')
    ax.set_aspect(1.0)