import os
import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms
def test_plot_manager_error(self, figure):
    with pytest.raises(AssertionError):
        with SimplePlottingAxes(ax=None, show=False, filename=None):
            raise AssertionError()