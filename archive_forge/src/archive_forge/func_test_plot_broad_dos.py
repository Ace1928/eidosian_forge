from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.mark.usefixtures('figure')
def test_plot_broad_dos(self, dense_dos, figure):
    ax = figure.add_subplot(111)
    _ = dense_dos.plot(ax=ax, npts=10, xmin=0, xmax=9, width=4, smearing='Gauss')
    line_data = ax.lines[0].get_data()
    assert np.allclose(line_data[0], range(10))
    assert np.allclose(line_data[1], [0.14659725, 0.19285644, 0.24345501, 0.29505574, 0.34335948, 0.38356488, 0.41104823, 0.42216901, 0.41503382, 0.39000808])