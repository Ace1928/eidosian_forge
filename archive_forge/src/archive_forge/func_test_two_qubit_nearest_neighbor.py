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
def test_two_qubit_nearest_neighbor(ax):
    value_map = {(grid_qubit.GridQubit(3, 2), grid_qubit.GridQubit(4, 2)): 0.004619111460557768, (grid_qubit.GridQubit(4, 1), grid_qubit.GridQubit(3, 2)): 0.0076079162393482835}
    with pytest.raises(ValueError, match='not nearest neighbors'):
        heatmap.TwoQubitInteractionHeatmap(value_map, coupler_width=0).plot(ax)