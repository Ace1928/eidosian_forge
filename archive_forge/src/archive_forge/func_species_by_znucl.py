from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
def species_by_znucl(structure: Structure) -> list[Species]:
    """
    Return list of unique specie found in structure **ordered according to sites**.

    Example:
        Site0: 0.5 0 0 O
        Site1: 0   0 0 Si

    produces [Specie_O, Specie_Si] and not set([Specie_O, Specie_Si]) as in `types_of_specie`
    """
    types = []
    for site in structure:
        for sp, v in site.species.items():
            if sp not in types and v != 0:
                types.append(sp)
    return types