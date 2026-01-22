import pytest
import numpy as np
from ase import Atom
from ase.build import bulk
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet

    Get energy from a LAMMPS calculation of an uncharged system.
    This was written to run with the 30 Apr 2019 version of LAMMPS,
    for which uncharged systems require the use of 'kspace_modify gewald'.
    