import numpy as np
import pytest
from matplotlib import lines, patches, text, spines, axis
from matplotlib import pyplot as plt
import cirq.testing
from cirq.vis.density_matrix import plot_density_matrix
from cirq.vis.density_matrix import _plot_element_of_density_matrix
@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('size', [2, 4, 8, 16])
def test_density_matrix_sizes_upper_bounds(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    ax = plot_density_matrix(matrix, show_text=show_text, title='Test Density Matrix Plot')
    circles = [c for c in ax.get_children() if isinstance(c, patches.Circle)]
    max_radius = np.max([c.radius for c in circles if c.fill])
    rects = [r for r in ax.get_children() if isinstance(r, patches.Rectangle) and r.get_alpha() is not None]
    max_height = np.max([r.get_height() for r in rects])
    max_width = np.max([r.get_width() for r in rects])
    assert max_height <= 1.0, "Some rectangle is exceeding out of it's cell size"
    assert max_width <= 1.0, "Some rectangle is exceeding out of it's cell size"
    assert max_radius * 2 <= 1.0, "Some circle is exceeding out of it's cell size"