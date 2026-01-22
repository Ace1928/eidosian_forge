import pathlib
import shutil
import string
from tempfile import mkdtemp
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from cirq.devices import grid_qubit
from cirq.vis import heatmap
def test_two_qubit_heatmap(ax):
    value_map = {(grid_qubit.GridQubit(3, 2), grid_qubit.GridQubit(4, 2)): 0.004619111460557768, (grid_qubit.GridQubit(4, 1), grid_qubit.GridQubit(4, 2)): 0.0076079162393482835}
    title = 'Two Qubit Interaction Heatmap'
    heatmap.TwoQubitInteractionHeatmap(value_map, title=title).plot(ax)
    assert ax.get_title() == title
    heatmap.TwoQubitInteractionHeatmap(value_map, title=title).plot()