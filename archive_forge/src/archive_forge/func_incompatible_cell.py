from math import sqrt
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import reference_states, atomic_numbers, chemical_symbols
from ase.utils import plural
def incompatible_cell(*, want, have):
    return RuntimeError('Cannot create {} cell for {} structure'.format(want, have))