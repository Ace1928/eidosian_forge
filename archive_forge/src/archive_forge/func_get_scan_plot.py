from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def get_scan_plot(self, coords=None):
    """
        Get a matplotlib plot of the potential energy surface.

        Args:
            coords: internal coordinate name to use as abscissa.
        """
    ax = pretty_plot(12, 8)
    dct = self.read_scan()
    if coords and coords in dct['coords']:
        x = dct['coords'][coords]
        ax.set_xlabel(coords)
    else:
        x = range(len(dct['energies']))
        ax.set_xlabel('points')
    ax.set_ylabel('Energy (eV)')
    e_min = min(dct['energies'])
    y = [(e - e_min) * Ha_to_eV for e in dct['energies']]
    ax.plot(x, y, 'ro--')
    return ax