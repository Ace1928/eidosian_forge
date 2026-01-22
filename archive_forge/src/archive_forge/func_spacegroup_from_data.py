import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def spacegroup_from_data(no=None, symbol=None, setting=None, centrosymmetric=None, scaled_primitive_cell=None, reciprocal_cell=None, subtrans=None, sitesym=None, rotations=None, translations=None, datafile=None):
    """Manually create a new space group instance.  This might be
    useful when reading crystal data with its own spacegroup
    definitions."""
    if no is not None and setting is not None:
        spg = Spacegroup(no, setting, datafile)
    elif symbol is not None:
        spg = Spacegroup(symbol, None, datafile)
    else:
        raise SpacegroupValueError('either *no* and *setting* or *symbol* must be given')
    if not isinstance(sitesym, list):
        raise TypeError('sitesym must be a list')
    have_sym = False
    if centrosymmetric is not None:
        spg._centrosymmetric = bool(centrosymmetric)
    if scaled_primitive_cell is not None:
        spg._scaled_primitive_cell = np.array(scaled_primitive_cell)
    if reciprocal_cell is not None:
        spg._reciprocal_cell = np.array(reciprocal_cell)
    if subtrans is not None:
        spg._subtrans = np.atleast_2d(subtrans)
        spg._nsubtrans = spg._subtrans.shape[0]
    if sitesym is not None:
        spg._rotations, spg._translations = parse_sitesym(sitesym)
        have_sym = True
    if rotations is not None:
        spg._rotations = np.atleast_3d(rotations)
        have_sym = True
    if translations is not None:
        spg._translations = np.atleast_2d(translations)
        have_sym = True
    if have_sym:
        if spg._rotations.shape[0] != spg._translations.shape[0]:
            raise SpacegroupValueError('inconsistent number of rotations and translations')
        spg._nsymop = spg._rotations.shape[0]
    return spg