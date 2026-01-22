import os
from warnings import warn
from glob import glob
import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.utils import writer
def read_states(states_dir):
    """Read structures stored by EON in the states directory *states_dir*."""
    subdirs = glob(os.path.join(states_dir, '[0123456789]*'))
    subdirs.sort(key=lambda d: int(os.path.basename(d)))
    images = [read_eon(os.path.join(subdir, 'reactant.con')) for subdir in subdirs]
    return images