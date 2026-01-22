from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.serialization import dumpfn, loadfn
from tqdm import tqdm
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine, Spin
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Orbital
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
def set_upper_lower_bands(self, e_lower, e_upper) -> None:
    """Set fake upper/lower bands, useful to set the same energy
        range in the spin up/down bands when calculating the DOS.
        """
    warnings.warn('This method does not work anymore in case of spin polarized case due to the concatenation of bands !')
    lower_band = e_lower * np.ones((1, self.ebands.shape[1]))
    upper_band = e_upper * np.ones((1, self.ebands.shape[1]))
    self.ebands = np.vstack((lower_band, self.ebands, upper_band))
    if self.proj:
        for sp, proj in self.proj.items():
            proj_lower = proj[:, 0:1, :, :]
            proj_upper = proj[:, -1:, :, :]
            self.proj[sp] = np.concatenate((proj_lower, proj, proj_upper), axis=1)