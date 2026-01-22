import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_set_current_figure_via_subfigure():
    fig1 = plt.figure()
    subfigs = fig1.subfigures(2)
    plt.figure()
    assert plt.gcf() != fig1
    current = plt.figure(subfigs[1])
    assert plt.gcf() == fig1
    assert current == fig1