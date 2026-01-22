import numpy as np
import pytest
import matplotlib.pyplot as plt
from cirq.vis import integrated_histogram
@pytest.mark.usefixtures('closefigures')
def test_multiple_plots():
    _, ax = plt.subplots(1, 1)
    n = 53
    data = np.random.random_sample((2, n))
    integrated_histogram(data[0], ax, color='r', label='data_1', median_line=False, mean_line=True, mean_label='mean_1')
    integrated_histogram(data[1], ax, color='k', label='data_2', median_label='median_2')
    assert ax.get_title() == 'N=53'
    for line in ax.get_lines():
        assert line.get_color() in ['r', 'k']
        assert line.get_label() in ['data_1', 'data_2', 'mean_1', 'median_2']