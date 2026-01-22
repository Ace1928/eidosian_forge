import numpy as np
import pytest
from matplotlib import lines, patches, text, spines, axis
from matplotlib import pyplot as plt
import cirq.testing
from cirq.vis.density_matrix import plot_density_matrix
from cirq.vis.density_matrix import _plot_element_of_density_matrix
@pytest.mark.parametrize('matrix', [np.random.random(size=(4, 4, 4)), np.random.random((3, 3)) * np.exp(np.random.random((3, 3)) * 2 * np.pi * 1j), np.random.random((4, 8)) * np.exp(np.random.random((4, 8)) * 2 * np.pi * 1j)])
def test_density_matrix_type_error(matrix):
    with pytest.raises(ValueError, match='Incorrect shape for density matrix:*'):
        plot_density_matrix(matrix)