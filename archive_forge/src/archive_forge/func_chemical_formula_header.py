import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
def chemical_formula_header(atoms):
    counts = atoms.symbols.formula.count()
    formula_sum = ' '.join((f'{sym}{count}' for sym, count in counts.items()))
    return f'_chemical_formula_structural       {atoms.symbols}\n_chemical_formula_sum              "{formula_sum}"\n'