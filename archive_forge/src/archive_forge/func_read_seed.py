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
def read_seed(seed, new_seed=None, ignore_internal_keys=False):
    """A wrapper around the CASTEP Calculator in conjunction with
    read_cell and read_param. Basically this can be used to reuse
    a previous calculation which results in a triple of
    cell/param/castep file. The label of the calculation if pre-
    fixed with `copy_of_` and everything else will be recycled as
    much as possible from the addressed calculation.

    Please note that this routine will return an atoms ordering as specified
    in the cell file! It will thus undo the potential reordering internally
    done by castep.
    """
    directory = os.path.abspath(os.path.dirname(seed))
    seed = os.path.basename(seed)
    paramfile = os.path.join(directory, '%s.param' % seed)
    cellfile = os.path.join(directory, '%s.cell' % seed)
    castepfile = os.path.join(directory, '%s.castep' % seed)
    checkfile = os.path.join(directory, '%s.check' % seed)
    atoms = read_cell(cellfile)
    atoms.calc._directory = directory
    atoms.calc._rename_existing_dir = False
    atoms.calc._castep_pp_path = directory
    atoms.calc.merge_param(paramfile, ignore_internal_keys=ignore_internal_keys)
    if new_seed is None:
        atoms.calc._label = 'copy_of_%s' % seed
    else:
        atoms.calc._label = str(new_seed)
    if os.path.isfile(castepfile):
        atoms.calc.read(castepfile)
        if os.path.isfile(checkfile):
            atoms.calc._check_file = os.path.basename(checkfile)
        atoms = atoms.calc.atoms
    else:
        pass
        warnings.warn('Corresponding *.castep file not found. Atoms object will be restored from *.cell and *.param only.')
    atoms.calc.push_oldstate()
    return atoms