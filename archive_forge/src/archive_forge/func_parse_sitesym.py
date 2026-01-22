import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def parse_sitesym(symlist, sep=',', force_positive_translation=False):
    """Parses a sequence of site symmetries in the form used by
    International Tables and returns corresponding rotation and
    translation arrays.

    Example:

    >>> symlist = [
    ...     'x,y,z',
    ...     '-y+1/2,x+1/2,z',
    ...     '-y,-x,-z',
    ...     'x-1/4, y-1/4, -z'
    ... ]
    >>> rot, trans = parse_sitesym(symlist)
    >>> rot
    array([[[ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]],
    <BLANKLINE>
           [[ 0, -1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1]],
    <BLANKLINE>
           [[ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0, -1]],
    <BLANKLINE>
           [[ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0, -1]]])
    >>> trans
    array([[ 0.  ,  0.  ,  0.  ],
           [ 0.5 ,  0.5 ,  0.  ],
           [ 0.  ,  0.  ,  0.  ],
           [-0.25, -0.25,  0.  ]])
    """
    nsym = len(symlist)
    rot = np.zeros((nsym, 3, 3), dtype='int')
    trans = np.zeros((nsym, 3))
    for i, sym in enumerate(symlist):
        parse_sitesym_single(sym, rot[i], trans[i], sep=sep, force_positive_translation=force_positive_translation)
    return (rot, trans)