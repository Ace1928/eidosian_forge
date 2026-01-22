from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from .util import pad_center, fill_off_diagonal, is_positive_int, tiny, expand_to
from .util.exceptions import ParameterError
from .filters import get_window
from typing import Any, Iterable, List, Optional, Tuple, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _IntLike_co
def dtw_backtracking(steps: np.ndarray, *, step_sizes_sigma: Optional[np.ndarray]=None, subseq: bool=False, start: Optional[Union[int, np.integer[Any]]]=None) -> np.ndarray:
    """Backtrack a warping path.

    Uses the saved step sizes from the cost accumulation
    step to backtrack the index pairs for a warping path.

    Parameters
    ----------
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.
    subseq : bool
        Enable subsequence DTW, e.g., for retrieval tasks.
    start : int
        Start column index for backtraing (only allowed for ``subseq=True``)

    Returns
    -------
    wp : list [shape=(N,)]
        Warping path with index pairs.
        Each list entry contains an index pair
        (n, m) as a tuple

    See Also
    --------
    dtw
    """
    if subseq is False and start is not None:
        raise ParameterError(f'start is only allowed to be set if subseq is True (start={start}, subseq={subseq})')
    default_steps = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.uint32)
    if step_sizes_sigma is None:
        step_sizes_sigma = default_steps
    else:
        step_sizes_sigma = np.concatenate((default_steps, step_sizes_sigma))
    wp = __dtw_backtracking(steps, step_sizes_sigma, subseq, start)
    return np.asarray(wp, dtype=int)