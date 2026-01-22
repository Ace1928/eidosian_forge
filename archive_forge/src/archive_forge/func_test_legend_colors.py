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
@pytest.mark.parametrize('color_type, param_dict, target', legend_color_tests, ids=legend_color_test_ids)
def test_legend_colors(color_type, param_dict, target):
    param_dict[f'legend.{color_type}color'] = param_dict.pop('color')
    get_func = f'get_{color_type}color'
    with mpl.rc_context(param_dict):
        _, ax = plt.subplots()
        ax.plot(range(3), label='test')
        leg = ax.legend()
        assert getattr(leg.legendPatch, get_func)() == target