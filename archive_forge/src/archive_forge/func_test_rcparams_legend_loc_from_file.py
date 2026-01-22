import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from cycler import cycler, Cycler
import pytest
import matplotlib as mpl
from matplotlib import _api, _c_internal_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.rcsetup import (
@pytest.mark.parametrize('value', ['best', 1, (0.9, 0.7), (-0.9, 0.7)])
def test_rcparams_legend_loc_from_file(tmpdir, value):
    rc_path = tmpdir.join('matplotlibrc')
    rc_path.write(f'legend.loc: {value}')
    with mpl.rc_context(fname=rc_path):
        assert mpl.rcParams['legend.loc'] == value