from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..scales.scale_discrete import scale_discrete
def assign_bins(x, breaks: FloatArrayLike, weight: Optional[FloatArrayLike]=None, pad: bool=False, closed: Literal['right', 'left']='right'):
    """
    Assign value in x to bins demacated by the break points

    Parameters
    ----------
    x :
        Values to be binned.
    breaks :
        Sequence of break points.
    weight :
        Weight of each value in `x`. Used in creating the frequency
        table. If `None`, then each value in `x` has a weight of 1.
    pad :
        If `True`, add empty bins at either end of `x`.
    closed :
        Whether the right or left edges of the bins are part of the
        bin.

    Returns
    -------
    out : dataframe
        Bin count and density information.
    """
    right = closed == 'right'
    if weight is None:
        weight = np.ones(len(x))
    else:
        weight = np.asarray(weight)
        weight[np.isnan(weight)] = 0
    bin_idx = pd.cut(x, bins=breaks, labels=False, right=right, include_lowest=True)
    bin_widths = np.diff(breaks)
    bin_x = (breaks[:-1] + breaks[1:]) * 0.5
    bins_long = pd.DataFrame({'bin_idx': bin_idx, 'weight': weight})
    wftable = bins_long.pivot_table('weight', index=['bin_idx'], aggfunc='sum')['weight']
    if len(wftable) < len(bin_x):
        empty_bins = set(range(len(bin_x))) - set(bin_idx)
        for b in empty_bins:
            wftable.loc[b] = 0
        wftable = wftable.sort_index()
    bin_count = wftable.tolist()
    if pad:
        bw0 = bin_widths[0]
        bwn = bin_widths[-1]
        bin_count = np.hstack([0, bin_count, 0])
        bin_widths = np.hstack([bw0, bin_widths, bwn])
        bin_x = np.hstack([bin_x[0] - bw0, bin_x, bin_x[-1] + bwn])
    return result_dataframe(bin_count, bin_x, bin_widths)