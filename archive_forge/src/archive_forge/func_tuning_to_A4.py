from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def tuning_to_A4(tuning: _ScalarOrSequence[_FloatLike_co], *, bins_per_octave: int=12) -> Union[np.floating[Any], np.ndarray]:
    """Convert a tuning deviation (from 0) in fractions of a bin per
    octave (e.g., ``tuning=-0.1``) to a reference pitch frequency
    relative to A440.

    This is useful if you are working in a non-A440 tuning system
    to determine the reference pitch frequency given a tuning
    offset and assuming equal temperament. By default, 12 bins per
    octave are used.

    This method is the inverse of  `A4_to_tuning`.

    Examples
    --------
    The base case of this method in which a tuning deviation of 0
    gets us to our A440 reference pitch.

    >>> librosa.tuning_to_A4(0.0)
    440.

    Convert a nonzero tuning offset to its reference pitch frequency.

    >>> librosa.tuning_to_A4(-0.318)
    431.992

    Convert 3 tuning deviations at once to respective reference
    pitch frequencies, using 36 bins per octave.

    >>> librosa.tuning_to_A4([0.1, 0.2, -0.1], bins_per_octave=36)
    array([   440.848,    441.698   439.154])

    Parameters
    ----------
    tuning : float or np.ndarray [shape=(n,), dtype=float]
        Tuning deviation from A440 in fractional bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    A4 : float or np.ndarray [shape=(n,), dtype=float]
        Reference frequency corresponding to A4.

    See Also
    --------
    A4_to_tuning
    """
    return 440.0 * 2.0 ** (np.asanyarray(tuning) / bins_per_octave)