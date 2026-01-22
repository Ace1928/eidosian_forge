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
@classmethod
def from_inputs(cls, inputs: list[BasicAbinitInput]) -> Self:
    """Build object from a list of BasicAbinitInput objects."""
    for inp in inputs:
        if any((p1 != p2 for p1, p2 in zip(inputs[0].pseudos, inp.pseudos))):
            raise ValueError('Pseudos must be consistent when from_inputs is invoked.')
    multi = cls(structure=[inp.structure for inp in inputs], pseudos=inputs[0].pseudos, ndtset=len(inputs))
    for inp, new_inp in zip(inputs, multi):
        new_inp.set_vars(**inp)
    return multi