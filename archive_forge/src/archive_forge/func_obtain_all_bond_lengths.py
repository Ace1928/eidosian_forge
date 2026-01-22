from __future__ import annotations
import collections
import json
import os
import warnings
from typing import TYPE_CHECKING
from pymatgen.core import Element
def obtain_all_bond_lengths(sp1, sp2, default_bl: float | None=None):
    """Obtain bond lengths for all bond orders from bond length database.

    Args:
        sp1 (Species): First specie.
        sp2 (Species): Second specie.
        default_bl: If a particular type of bond does not exist, use this
            bond length as a default value (bond order = 1).
            If None, a ValueError will be thrown.

    Returns:
        A dict mapping bond order to bond length in angstrom
    """
    if isinstance(sp1, Element):
        sp1 = sp1.symbol
    if isinstance(sp2, Element):
        sp2 = sp2.symbol
    syms = tuple(sorted([sp1, sp2]))
    if syms in bond_lengths:
        return bond_lengths[syms].copy()
    if default_bl is not None:
        return {1: default_bl}
    raise ValueError(f'No bond data for elements {syms[0]} - {syms[1]}')