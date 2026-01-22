from __future__ import annotations
import collections
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.util.coord import pbc_diff
@property
def species_string(self) -> str:
    """String representation of species on the site."""
    if self.is_ordered:
        return str(next(iter(self.species)))
    return ', '.join((f'{sp}:{self.species[sp]:.3}' for sp in sorted(self.species)))