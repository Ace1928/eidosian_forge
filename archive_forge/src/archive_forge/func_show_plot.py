from __future__ import annotations
import abc
import collections
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.spectrum import Spectrum
from pymatgen.util.plotting import add_fig_kwargs, pretty_plot
def show_plot(self, structure: Structure, **kwargs):
    """
        Shows the diffraction plot.

        Args:
            structure (Structure): Input structure
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
            annotate_peaks (str | None): Whether and how to annotate the peaks
                with hkl indices. Default is 'compact', i.e. show short
                version (oriented vertically), e.g. 100. If 'full', show
                long version, e.g. (1, 0, 0). If None, do not show anything.
        """
    self.get_plot(structure, **kwargs).show()