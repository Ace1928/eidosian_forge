from __future__ import annotations
import typing
from copy import copy
import numpy as np
from ..iapi import panel_ranges
def munch_data(data: pd.DataFrame, dist: FloatArray) -> pd.DataFrame:
    """
    Breakup path into small segments
    """
    x, y = (data['x'], data['y'])
    segment_length = 0.01
    dist[np.isnan(dist)] = 1
    extra = np.maximum(np.floor(dist / segment_length), 1)
    extra = extra.astype(int, copy=False)
    x = [interp(start, end, n) for start, end, n in zip(x[:-1], x[1:], extra)]
    y = [interp(start, end, n) for start, end, n in zip(y[:-1], y[1:], extra)]
    x.append(data['x'].iloc[-1])
    y.append(data['y'].iloc[-1])
    x = np.hstack(x)
    y = np.hstack(y)
    idx = np.hstack([np.repeat(data.index[:-1], extra), len(data) - 1])
    munched = data.loc[idx, list(data.columns.difference(['x', 'y']))]
    munched['x'] = x
    munched['y'] = y
    munched.reset_index(drop=True, inplace=True)
    return munched