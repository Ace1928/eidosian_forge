import collections
from math import sin, pi, sqrt
from numbers import Real, Integral
from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.atoms import Atoms
import ase.units as units
import ase.io
from ase.utils import jsonable, lazymethod
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spectrum.dosdata import RawDOSData
from ase.spectrum.doscollection import DOSCollection
def get_energies_and_modes(self, all_atoms: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalise the Hessian to obtain harmonic modes

        Results are cached so diagonalization will only be performed once for
        this object instance.

        Args:
            all_atoms:
                If True, return modes as (3N, [N + N_frozen], 3) array where
                the second axis corresponds to the full list of atoms in the
                attached atoms object. Atoms that were not included in the
                Hessian will have displacement vectors of (0, 0, 0).

        Returns:
            tuple (energies, modes)

            Energies are given in units of eV. (To convert these to frequencies
            in cm-1, divide by ase.units.invcm.)

            Modes are given in Cartesian coordinates as a (3N, N, 3) array
            where indices correspond to the (mode_index, atom, direction).

        """
    energies, modes_from_hessian = self._energies_and_modes()
    if all_atoms:
        n_active_atoms = len(self.get_indices())
        n_all_atoms = len(self._atoms)
        modes = np.zeros((3 * n_active_atoms, n_all_atoms, 3))
        modes[:, self.get_mask(), :] = modes_from_hessian
    else:
        modes = modes_from_hessian.copy()
    return (energies.copy(), modes)