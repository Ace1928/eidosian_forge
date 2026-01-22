from __future__ import annotations
import itertools
import warnings
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Composition, DummySpecies, Element, Lattice, Molecule, Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
def to_positions(self) -> None:
    """Convert displacements between consecutive frames into positions.

        `base_positions` and `coords` should both be in fractional coords or
        absolute coords.

        This is the opposite operation of `to_displacements()`.
        """
    if not self.coords_are_displacement:
        return
    cumulative_displacements = np.cumsum(self.coords, axis=0)
    positions = self.base_positions + cumulative_displacements
    self.coords = positions
    self.coords_are_displacement = False