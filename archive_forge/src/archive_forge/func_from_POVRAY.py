from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
@classmethod
def from_POVRAY(cls, povray, density_grid, cut_off, **kwargs):
    return cls(cell=povray.cell, cell_origin=povray.cell_vertices[0, 0, 0], density_grid=density_grid, cut_off=cut_off, **kwargs)