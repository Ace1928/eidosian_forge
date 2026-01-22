from typing import Dict, List, Tuple, Union, Optional
from numbers import Real
from collections import namedtuple
import re
from string import digits
import numpy as np
from ase import Atoms
from ase.units import Angstrom, Bohr, nm
def set_defs(self, defs: Union[Dict[str, float], str, List[str], None]) -> None:
    self.defs = dict()
    if defs is None:
        return
    if isinstance(defs, dict):
        self.defs.update(**defs)
        return
    if isinstance(defs, str):
        defs = _re_linesplit.split(defs.strip())
    for row in defs:
        key, val = _re_defs.split(row)
        self.defs[key] = self.get_var(val)