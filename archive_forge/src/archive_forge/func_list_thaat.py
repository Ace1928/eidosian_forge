import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
def list_thaat() -> List[str]:
    """List supported thaats by name.

    Returns
    -------
    thaats : list
        A list of supported thaats

    Examples
    --------
    >>> librosa.list_thaat()
    ['bilaval',
     'khamaj',
     'kafi',
     'asavari',
     'bhairavi',
     'kalyan',
     'marva',
     'poorvi',
     'todi',
     'bhairav']

    See Also
    --------
    list_mela
    thaat_to_degrees
    """
    return list(THAAT_MAP.keys())