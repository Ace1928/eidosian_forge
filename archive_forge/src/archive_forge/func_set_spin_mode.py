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
def set_spin_mode(self, spin_mode):
    """
        Set the variables used to the treat the spin degree of freedom.
        Return dictionary with the variables that have been removed.

        Args:
            spin_mode: SpinMode object or string. Possible values for string are:

            - polarized
            - unpolarized
            - afm (anti-ferromagnetic)
            - spinor (non-collinear magnetism)
            - spinor_nomag (non-collinear, no magnetism)
        """
    old_vars = self.pop_vars(['nsppol', 'nspden', 'nspinor'])
    self.add_abiobjects(aobj.SpinMode.as_spinmode(spin_mode))
    return old_vars