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
@pytest.mark.parametrize('tuple_keys', [True, False])
def test_cells_positions(ax, tuple_keys):
    row_col_list = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    qubits = [grid_qubit.GridQubit(row, col) for row, col in row_col_list]
    values = np.random.random(len(qubits))
    test_value_map = {(qubit,) if tuple_keys else qubit: value for qubit, value in zip(qubits, values)}
    _, collection = heatmap.Heatmap(test_value_map).plot(ax)
    found_qubits = set()
    for path in collection.get_paths():
        vertices = path.vertices[0:4]
        row = int(round(np.mean([v[1] for v in vertices])))
        col = int(round(np.mean([v[0] for v in vertices])))
        found_qubits.add((row, col))
    assert found_qubits == set(row_col_list)