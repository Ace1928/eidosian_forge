import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_switch_backend_no_close():
    plt.switch_backend('agg')
    fig = plt.figure()
    fig = plt.figure()
    assert len(plt.get_fignums()) == 2
    plt.switch_backend('agg')
    assert len(plt.get_fignums()) == 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        plt.switch_backend('svg')
    assert len(plt.get_fignums()) == 0