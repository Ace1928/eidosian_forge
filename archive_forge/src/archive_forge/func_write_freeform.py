import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def write_freeform(fd, outputobj):
    """
    Prints out to a given file a CastepInputFile or derived class, such as
    CastepCell or CastepParam.
    """
    options = outputobj._options
    preferred_order = ['lattice_cart', 'lattice_abc', 'positions_frac', 'positions_abs', 'species_pot', 'symmetry_ops', 'task', 'cut_off_energy']
    keys = outputobj.get_attr_dict().keys()
    keys = sorted(keys, key=lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order))
    for kw in keys:
        opt = options[kw]
        if opt.type.lower() == 'block':
            fd.write('%BLOCK {0}\n{1}\n%ENDBLOCK {0}\n\n'.format(kw.upper(), opt.value.strip('\n')))
        else:
            fd.write('{0}: {1}\n'.format(kw.upper(), opt.value))