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
@pytest.mark.usefixtures('closefigures')
def test_plot_updates_local_config():
    value_map_2d = {(grid_qubit.GridQubit(3, 2), grid_qubit.GridQubit(4, 2)): 0.004619111460557768, (grid_qubit.GridQubit(4, 1), grid_qubit.GridQubit(4, 2)): 0.0076079162393482835}
    value_map_1d = {(grid_qubit.GridQubit(3, 2),): 0.004619111460557768, (grid_qubit.GridQubit(4, 2),): 0.0076079162393482835}
    original_title = 'Two Qubit Interaction Heatmap'
    new_title = 'Temporary title for the plot'
    for random_heatmap in [heatmap.TwoQubitInteractionHeatmap(value_map_2d, title=original_title), heatmap.Heatmap(value_map_1d, title=original_title)]:
        _, ax = plt.subplots()
        random_heatmap.plot(ax, title=new_title)
        assert ax.get_title() == new_title
        _, ax = plt.subplots()
        random_heatmap.plot(ax)
        assert ax.get_title() == original_title