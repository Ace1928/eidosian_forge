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
def get_partial_doses(self, tdos, eband_ud, spins, enr, npts_mu, T, progress):
    """Return a CompleteDos object interpolating the projections.

        tdos: total dos previously calculated
        npts_mu: number of energy points of the Dos
        T: parameter used to smooth the Dos
        progress: Default False, If True a progress bar is shown.
        """
    if not self.data.proj:
        raise BoltztrapError('No projections loaded.')
    bkp_data_ebands = np.copy(self.data.ebands)
    pdoss = {}
    if progress:
        n_iter = np.prod(np.sum([np.array(i.shape)[2:] for i in self.data.proj.values()]))
        t = tqdm(total=n_iter * 2)
    for spin, eb in zip(spins, eband_ud):
        for idx, site in enumerate(self.data.structure):
            if site not in pdoss:
                pdoss[site] = {}
            for iorb, orb in enumerate(Orbital):
                if progress:
                    t.update()
                if iorb == self.data.proj[spin].shape[-1]:
                    break
                if orb not in pdoss[site]:
                    pdoss[site][orb] = {}
                self.data.ebands = self.data.proj[spin][:, :, idx, iorb].T
                coeffs = fite.fitde3D(self.data, self.equivalences)
                proj, _vvproj, _cproj = fite.getBTPbands(self.equivalences, coeffs, self.data.lattvec)
                edos, pdos = BL.DOS(eb, npts=npts_mu, weights=np.abs(proj.real), erange=enr)
                if T:
                    pdos = BL.smoothen_DOS(edos, pdos, T)
                pdoss[site][orb][spin] = pdos
    self.data.ebands = bkp_data_ebands
    return CompleteDos(self.data.structure, total_dos=tdos, pdoss=pdoss)