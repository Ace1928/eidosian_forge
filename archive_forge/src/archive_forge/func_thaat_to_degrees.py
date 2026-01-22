import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
def thaat_to_degrees(thaat: str) -> np.ndarray:
    """Construct the svara indices (degrees) for a given thaat

    Parameters
    ----------
    thaat : str
        The name of the thaat

    Returns
    -------
    indices : np.ndarray
        A list of the seven svara indices (starting from 0=Sa)
        contained in the specified thaat

    See Also
    --------
    key_to_degrees
    mela_to_degrees
    list_thaat

    Examples
    --------
    >>> librosa.thaat_to_degrees('bilaval')
    array([ 0,  2,  4,  5,  7,  9, 11])

    >>> librosa.thaat_to_degrees('todi')
    array([ 0,  1,  3,  6,  7,  8, 11])
    """
    return np.asarray(THAAT_MAP[thaat.lower()])