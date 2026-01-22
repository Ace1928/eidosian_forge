import numpy as np
import numba
from .exceptions import ParameterError
from .utils import valid_intervals
from .._typing import _SequenceLike
def match_intervals(intervals_from: np.ndarray, intervals_to: np.ndarray, strict: bool=True) -> np.ndarray:
    """Match one set of time intervals to another.

    This can be useful for tasks such as mapping beat timings
    to segments.

    Each element ``[a, b]`` of ``intervals_from`` is matched to the
    element ``[c, d]`` of ``intervals_to`` which maximizes the
    Jaccard similarity between the intervals::

        max(0, |min(b, d) - max(a, c)|) / |max(d, b) - min(a, c)|

    In ``strict=True`` mode, if there is no interval with positive
    intersection with ``[a,b]``, an exception is thrown.

    In ``strict=False`` mode, any interval ``[a, b]`` that has no
    intersection with any element of ``intervals_to`` is instead
    matched to the interval ``[c, d]`` which minimizes::

        min(|b - c|, |a - d|)

    that is, the disjoint interval [c, d] with a boundary closest
    to [a, b].

    .. note:: An element of ``intervals_to`` may be matched to multiple
       entries of ``intervals_from``.

    Parameters
    ----------
    intervals_from : np.ndarray [shape=(n, 2)]
        The time range for source intervals.
        The ``i`` th interval spans time ``intervals_from[i, 0]``
        to ``intervals_from[i, 1]``.
        ``intervals_from[0, 0]`` should be 0, ``intervals_from[-1, 1]``
        should be the track duration.
    intervals_to : np.ndarray [shape=(m, 2)]
        Analogous to ``intervals_from``.
    strict : bool
        If ``True``, intervals can only match if they intersect.
        If ``False``, disjoint intervals can match.

    Returns
    -------
    interval_mapping : np.ndarray [shape=(n,)]
        For each interval in ``intervals_from``, the
        corresponding interval in ``intervals_to``.

    See Also
    --------
    match_events

    Raises
    ------
    ParameterError
        If either array of input intervals is not the correct shape

        If ``strict=True`` and some element of ``intervals_from`` is disjoint from
        every element of ``intervals_to``.

    Examples
    --------
    >>> ints_from = np.array([[3, 5], [1, 4], [4, 5]])
    >>> ints_to = np.array([[0, 2], [1, 3], [4, 5], [6, 7]])
    >>> librosa.util.match_intervals(ints_from, ints_to)
    array([2, 1, 2], dtype=uint32)
    >>> # [3, 5] => [4, 5]  (ints_to[2])
    >>> # [1, 4] => [1, 3]  (ints_to[1])
    >>> # [4, 5] => [4, 5]  (ints_to[2])

    The reverse matching of the above is not possible in ``strict`` mode
    because ``[6, 7]`` is disjoint from all intervals in ``ints_from``.
    With ``strict=False``, we get the following:

    >>> librosa.util.match_intervals(ints_to, ints_from, strict=False)
    array([1, 1, 2, 2], dtype=uint32)
    >>> # [0, 2] => [1, 4]  (ints_from[1])
    >>> # [1, 3] => [1, 4]  (ints_from[1])
    >>> # [4, 5] => [4, 5]  (ints_from[2])
    >>> # [6, 7] => [4, 5]  (ints_from[2])
    """
    if len(intervals_from) == 0 or len(intervals_to) == 0:
        raise ParameterError('Attempting to match empty interval list')
    valid_intervals(intervals_from)
    valid_intervals(intervals_to)
    try:
        return __match_intervals(intervals_from, intervals_to, strict=strict)
    except ParameterError as exc:
        raise ParameterError(f'Unable to match intervals with strict={strict}') from exc