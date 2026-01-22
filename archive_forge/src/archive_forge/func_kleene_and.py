from __future__ import annotations
import numpy as np
from pandas._libs import (
def kleene_and(left: bool | libmissing.NAType | np.ndarray, right: bool | libmissing.NAType | np.ndarray, left_mask: np.ndarray | None, right_mask: np.ndarray | None):
    """
    Boolean ``and`` using Kleene logic.

    Values are ``NA`` for ``NA & NA`` or ``True & NA``.

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical xor, and the new mask.
    """
    if left_mask is None:
        return kleene_and(right, left, right_mask, left_mask)
    if not isinstance(left, np.ndarray):
        raise TypeError('Either `left` or `right` need to be a np.ndarray.')
    raise_for_nan(right, method='and')
    if right is libmissing.NA:
        result = np.zeros_like(left)
    else:
        result = left & right
    if right_mask is None:
        if right is libmissing.NA:
            mask = left & ~left_mask | left_mask
        else:
            mask = left_mask.copy()
            if right is False:
                mask[:] = False
    else:
        left_false = ~(left | left_mask)
        right_false = ~(right | right_mask)
        mask = left_mask & ~right_false | right_mask & ~left_false
    return (result, mask)