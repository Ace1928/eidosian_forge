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
@mpl.style.context('default')
@check_figures_equal(extensions=['png'])
def test_pickle_load_from_subprocess(fig_test, fig_ref, tmp_path):
    _generate_complete_test_figure(fig_ref)
    fp = tmp_path / 'sinus.pickle'
    assert not fp.exists()
    with fp.open('wb') as file:
        pickle.dump(fig_ref, file, pickle.HIGHEST_PROTOCOL)
    assert fp.exists()
    proc = subprocess_run_helper(_pickle_load_subprocess, timeout=60, extra_env={'PICKLE_FILE_PATH': str(fp), 'MPLBACKEND': 'Agg'})
    loaded_fig = pickle.loads(ast.literal_eval(proc.stdout))
    loaded_fig.canvas.draw()
    fig_test.set_size_inches(loaded_fig.get_size_inches())
    fig_test.figimage(loaded_fig.canvas.renderer.buffer_rgba())
    plt.close(loaded_fig)