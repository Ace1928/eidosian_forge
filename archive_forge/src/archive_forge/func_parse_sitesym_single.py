import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def parse_sitesym_single(sym, out_rot, out_trans, sep=',', force_positive_translation=False):
    """Parses a single site symmetry in the form used by International 
    Tables and overwrites 'out_rot' and 'out_trans' with data.
    
    Parameters
    ----------
    
    sym: str
      Site symmetry in the form used by International Tables (e.g. "x,y,z", "y-1/2,x,-z").
      
    out_rot: np.array
      A 3x3-integer array representing rotations (changes are made inplace).
      
    out_rot: np.array
      A 3-float array representing translations (changes are made inplace).
      
    sep: str
      String separator ("," in "x,y,z").
      
    force_positive_translation: bool
      Forces fractional translations to be between 0 and 1 (otherwise negative values might be accepted).
      Defaults to 'False'.
      
      
    Returns
    -------
    
    Nothing is returned: 'out_rot' and 'out_trans' are changed inplace.
    
    
    """
    out_rot[:] = 0.0
    out_trans[:] = 0.0
    for i, element in enumerate(sym.split(sep)):
        e_rot_list, e_trans = parse_sitesym_element(element)
        for rot_idx, rot_sgn in e_rot_list:
            out_rot[i][rot_idx] = rot_sgn
        out_trans[i] = e_trans % 1.0 if force_positive_translation else e_trans