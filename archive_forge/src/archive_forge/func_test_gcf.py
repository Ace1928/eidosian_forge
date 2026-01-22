from io import BytesIO
import ast
import pickle
import pickletools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import cm
from matplotlib.testing import subprocess_run_helper
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.dates import rrulewrapper
from matplotlib.lines import VertexSelector
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.figure as mfigure
from mpl_toolkits.axes_grid1 import parasite_axes  # type: ignore
def test_gcf():
    fig = plt.figure('a label')
    buf = BytesIO()
    pickle.dump(fig, buf, pickle.HIGHEST_PROTOCOL)
    plt.close('all')
    assert plt._pylab_helpers.Gcf.figs == {}
    fig = pickle.loads(buf.getbuffer())
    assert plt._pylab_helpers.Gcf.figs != {}
    assert fig.get_label() == 'a label'