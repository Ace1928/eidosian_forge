import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def print_symmetry(symprec, dataset):
    print('ase.spacegroup.symmetrize: prec', symprec, 'got symmetry group number', dataset['number'], ', international (Hermann-Mauguin)', dataset['international'], ', Hall ', dataset['hall'])