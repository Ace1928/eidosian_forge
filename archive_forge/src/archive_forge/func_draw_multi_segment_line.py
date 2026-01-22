import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def draw_multi_segment_line(cvs, points, antialias):
    """Draw multi-line segment line.

    Parameters
    ----------
    cvs: canvas
      A Datashader canvas
    points: list of tuples
      List of tuples of two scalars that represent each of the vertices in the
      multi-segment line.
    antialias: boolean
      To anti-alias or not is the question

    Returns
    -------
    agg: A Datashader aggregator (xarray)
    """
    x, y = ([], [])
    for x1, y1 in points:
        x.append(x1)
        y.append(y1)
    xs, ys = (np.array(x), np.array(y))
    points = pd.DataFrame({'x': xs, 'y': ys, 'val': 5.0})
    agg = cvs.line(points, 'x', 'y', agg=ds.reductions.max('val'), antialias=antialias)
    return xr.concat([agg], 'stack').sum(dim='stack')