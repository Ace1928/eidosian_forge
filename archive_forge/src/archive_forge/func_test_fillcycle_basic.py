import contextlib
from io import StringIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cycler import cycler
def test_fillcycle_basic():
    fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('c', ['r', 'g', 'y']) + cycler('hatch', ['xx', 'O', '|-']) + cycler('linestyle', ['-', '--', ':']))
    for _ in range(4):
        ax.fill(range(10), range(10))
    assert [p.get_facecolor() for p in ax.patches] == [mpl.colors.to_rgba(c) for c in ['r', 'g', 'y', 'r']]
    assert [p.get_hatch() for p in ax.patches] == ['xx', 'O', '|-', 'xx']
    assert [p.get_linestyle() for p in ax.patches] == ['-', '--', ':', '-']