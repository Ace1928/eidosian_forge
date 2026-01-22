from __future__ import annotations
import collections
import json
import os
import warnings
from typing import TYPE_CHECKING
from pymatgen.core import Element
def get_bond_length(sp1: SpeciesLike, sp2: SpeciesLike, bond_order: float=1) -> float:
    """Get the bond length between two species.

    Args:
        sp1 (Species): First specie.
        sp2 (Species): Second specie.
        bond_order: For species with different possible bond orders,
            this allows one to obtain the bond length for a particular bond
            order. For example, to get the C=C bond length instead of the
            C-C bond length, this should be set to 2. Defaults to 1.

    Returns:
        Bond length in Angstrom. If no data is available, the sum of the atomic
        radius is used.
    """
    sp1 = Element(sp1) if isinstance(sp1, str) else sp1
    sp2 = Element(sp2) if isinstance(sp2, str) else sp2
    try:
        all_lengths = obtain_all_bond_lengths(sp1, sp2)
        return all_lengths[bond_order]
    except (ValueError, KeyError):
        warnings.warn(f'No order {bond_order} bond lengths between {sp1} and {sp2} found in database. Returning sum of atomic radius.')
        return sp1.atomic_radius + sp2.atomic_radius