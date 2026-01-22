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
def has_structure(self):
    """Whether this CIF block has an atomic configuration."""
    try:
        self.get_symbols()
        self._get_site_coordinates()
    except NoStructureData:
        return False
    else:
        return True