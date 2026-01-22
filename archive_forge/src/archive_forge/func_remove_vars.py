from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
def remove_vars(self, keys: Sequence[str], strict: bool=True) -> dict[str, InputVariable]:
    """
        Remove the variables listed in keys.
        Return dictionary with the variables that have been removed.

        Args:
            keys: string or list of strings with variable names.
            strict: If True, KeyError is raised if at least one variable is not present.
        """
    if isinstance(keys, str):
        keys = [keys]
    removed = {}
    for key in keys:
        if strict and key not in self:
            raise KeyError(f'key={key!r} not in self:\n {list(self)}')
        if key in self:
            removed[key] = self.pop(key)
    return removed