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
def get_modes(self, all_atoms: bool=False) -> np.ndarray:
    """Diagonalise the Hessian to obtain harmonic modes

        Results are cached so diagonalization will only be performed once for
        this object instance.

        all_atoms:
            If True, return modes as (3N, [N + N_frozen], 3) array where
            the second axis corresponds to the full list of atoms in the
            attached atoms object. Atoms that were not included in the
            Hessian will have displacement vectors of (0, 0, 0).

        Returns:
            Modes in Cartesian coordinates as a (3N, N, 3) array where indices
            correspond to the (mode_index, atom, direction).

        """
    return self.get_energies_and_modes(all_atoms=all_atoms)[1]