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
@pytest.mark.parametrize('format_string', ['.3e', '.2f', '.4g', 's'])
def test_non_float_values(ax, format_string):

    class Foo:

        def __init__(self, value: float, unit: str):
            self.value = value
            self.unit = unit

        def __float__(self):
            return self.value

        def __format__(self, format_string):
            if format_string == 's':
                return f'{self.value}{self.unit}'
            else:
                return format(self.value, format_string)
    row_col_list = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    qubits = [grid_qubit.GridQubit(row, col) for row, col in row_col_list]
    values = np.random.random(len(qubits))
    units = np.random.choice([c for c in string.ascii_letters], len(qubits))
    test_value_map = {(qubit,): Foo(float(value), unit) for qubit, value, unit in zip(qubits, values, units)}
    row_col_map = {row_col: Foo(float(value), unit) for row_col, value, unit in zip(row_col_list, values, units)}
    colormap_name = 'viridis'
    vmin, vmax = (0.0, 1.0)
    random_heatmap = heatmap.Heatmap(test_value_map, collection_options={'cmap': colormap_name}, vmin=vmin, vmax=vmax, annotation_format=format_string)
    _, mesh = random_heatmap.plot(ax)
    colormap = mpl.colormaps[colormap_name]
    for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
        vertices = path.vertices[0:4]
        row = int(round(np.mean([v[1] for v in vertices])))
        col = int(round(np.mean([v[0] for v in vertices])))
        foo = row_col_map[row, col]
        color_scale = (foo.value - vmin) / (vmax - vmin)
        expected_color = np.array(colormap(color_scale))
        assert np.all(np.isclose(facecolor, expected_color))
    for artist in ax.get_children():
        if isinstance(artist, mpl.text.Text):
            col, row = artist.get_position()
            if (row, col) in test_value_map:
                foo = test_value_map[row, col]
                actual_text = artist.get_text()
                expected_text = format(foo, format_string)
                assert actual_text == expected_text