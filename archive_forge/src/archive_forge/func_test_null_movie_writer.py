import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
def test_null_movie_writer(anim):
    plt.rcParams['savefig.facecolor'] = 'auto'
    filename = 'unused.null'
    dpi = 50
    savefig_kwargs = dict(foo=0)
    writer = NullMovieWriter()
    anim.save(filename, dpi=dpi, writer=writer, savefig_kwargs=savefig_kwargs)
    assert writer.fig == plt.figure(1)
    assert writer.outfile == filename
    assert writer.dpi == dpi
    assert writer.args == ()
    for k, v in savefig_kwargs.items():
        assert writer.savefig_kwargs[k] == v
    assert writer._count == anim._save_count