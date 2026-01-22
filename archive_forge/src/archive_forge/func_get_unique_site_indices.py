from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
def get_unique_site_indices(struct: Structure | Molecule):
    """
    Get unique site indices for a structure according to site properties. Whatever site-property
    has the most unique values is used for indexing.

    For example, if you have magnetic CoO with half Co atoms having a positive moment, and the
    other half having a negative moment. Then this function will create a dict of sites for
    Co_1, Co_2, O. This function also deals with "Species" properties like oxi_state and spin by
    pushing them to site properties.

    This creates unique sites, based on site properties, but does not have anything to do with
    turning those site properties into CP2K input parameters. This will only be done for properties
    which can be turned into CP2K input parameters, which are stored in parsable_site_properties.
    """
    spins = []
    oxi_states = []
    parsable_site_properties = {'magmom', 'oxi_state', 'spin', 'u_minus_j', 'basis', 'potential', 'ghost', 'aux_basis'}
    for site in struct:
        for sp in site.species:
            oxi_states.append(getattr(sp, 'oxi_state', 0))
            spins.append(getattr(sp, '_properties', {}).get('spin', 0))
    struct.add_site_property('oxi_state', oxi_states)
    struct.add_site_property('spin', spins)
    struct.remove_oxidation_states()
    items = [(site.species_string, *[struct.site_properties[k][idx] for k in struct.site_properties if k.lower() in parsable_site_properties]) for idx, site in enumerate(struct)]
    unique_itms = list(set(items))
    _sites: dict[tuple, list] = {u: [] for u in unique_itms}
    for i, itm in enumerate(items):
        _sites[itm].append(i)
    sites = {}
    nums = dict.fromkeys(struct.symbol_set, 1)
    for s in _sites:
        sites[f'{s[0]}_{nums[s[0]]}'] = _sites[s]
        nums[s[0]] += 1
    return sites