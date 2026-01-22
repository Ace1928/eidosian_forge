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
@pytest.mark.parametrize('colormap_name', ['viridis', 'Greys', 'binary', 'PiYG', 'twilight', 'Pastel1', 'flag'])
def test_cell_colors(ax, colormap_name):
    row_col_list = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    qubits = [grid_qubit.GridQubit(row, col) for row, col in row_col_list]
    values = 1.0 + 2.0 * np.random.random(len(qubits))
    test_value_map = {(qubit,): value for qubit, value in zip(qubits, values)}
    test_row_col_map = {rc: value for rc, value in zip(row_col_list, values)}
    vmin, vmax = (1.5, 2.5)
    random_heatmap = heatmap.Heatmap(test_value_map, collection_options={'cmap': colormap_name}, vmin=vmin, vmax=vmax)
    _, mesh = random_heatmap.plot(ax)
    colormap = mpl.colormaps[colormap_name]
    for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
        vertices = path.vertices[0:4]
        row = int(round(np.mean([v[1] for v in vertices])))
        col = int(round(np.mean([v[0] for v in vertices])))
        value = test_row_col_map[row, col]
        color_scale = (value - vmin) / (vmax - vmin)
        color_scale = max(color_scale, 0.0)
        color_scale = min(color_scale, 1.0)
        expected_color = np.array(colormap(color_scale))
        assert np.all(np.isclose(facecolor, expected_color))