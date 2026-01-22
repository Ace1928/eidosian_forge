import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
def list_mela() -> Dict[str, int]:
    """List melakarta ragas by name and index.

    Melakarta raga names are transcribed from [#]_, with the exception of #45
    (subhapanthuvarali).

    .. [#] Bhagyalekshmy, S. (1990).
        Ragas in Carnatic music.
        South Asia Books.

    Returns
    -------
    mela_map : dict
        A dictionary mapping melakarta raga names to indices (1, 2, ..., 72)

    Examples
    --------
    >>> librosa.list_mela()
    {'kanakangi': 1,
     'ratnangi': 2,
     'ganamurthi': 3,
     'vanaspathi': 4,
     ...}

    See Also
    --------
    mela_to_degrees
    mela_to_svara
    list_thaat
    """
    return MELAKARTA_MAP.copy()