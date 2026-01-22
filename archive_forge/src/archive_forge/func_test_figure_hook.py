import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_figure_hook():
    test_rc = {'figure.hooks': ['matplotlib.tests.test_pyplot:figure_hook_example']}
    with mpl.rc_context(test_rc):
        fig = plt.figure()
    assert fig._test_was_here