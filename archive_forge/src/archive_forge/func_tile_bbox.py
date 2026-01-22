from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
def tile_bbox(self, x, y, z, y0_at_north_pole=True):
    """
        Return the ``(x0, x1), (y0, y1)`` bounding box for the given x, y, z
        tile position.

        Parameters
        ----------
        x
            The x tile coordinate in the Google tile numbering system.
        y
            The y tile coordinate in the Google tile numbering system.
        z
            The z tile coordinate in the Google tile numbering system.

        y0_at_north_pole: optional
            Boolean representing whether the numbering of the y coordinate
            starts at the north pole (as is the convention for Google tiles)
            or not (in which case it will start at the south pole, as is the
            convention for TMS). Defaults to True.


        """
    n = 2 ** z
    assert 0 <= x <= n - 1, f"Tile's x index is out of range. Upper limit {n}. Got {x}"
    assert 0 <= y <= n - 1, f"Tile's y index is out of range. Upper limit {n}. Got {y}"
    x0, x1 = self.crs.x_limits
    y0, y1 = self.crs.y_limits
    box_h = (y1 - y0) / n
    box_w = (x1 - x0) / n
    n_xs = x0 + (x + np.arange(0, 2, dtype=np.float64)) * box_w
    n_ys = y0 + (y + np.arange(0, 2, dtype=np.float64)) * box_h
    if y0_at_north_pole:
        n_ys = -1 * n_ys[::-1]
    return (n_xs, n_ys)