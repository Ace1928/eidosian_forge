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
def bandana(self, emin=-np.inf, emax=np.inf):
    """Cut out bands outside the range (emin,emax)."""
    band_min = np.min(self.ebands, axis=1)
    band_max = np.max(self.ebands, axis=1)
    ii = np.nonzero(band_min < emax)
    n_emax = ii[0][-1]
    ii = np.nonzero(band_max > emin)
    n_emin = ii[0][0]
    self.ebands = self.ebands[n_emin:n_emax + 1]
    if isinstance(self.proj, np.ndarray):
        self.proj = self.proj[:, n_emin:n_emax + 1, :, :]
    if self.mommat is not None:
        self.mommat = self.mommat[:, n_emin:n_emax + 1, :]
    if self.nelect is not None:
        self.nelect -= self.dosweight * n_emin
    return (n_emin, n_emax)