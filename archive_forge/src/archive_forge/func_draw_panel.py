from __future__ import annotations
import typing
from warnings import warn
import numpy as np
from .._utils import resolution
from ..coords import coord_cartesian
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from .geom import geom
from .geom_polygon import geom_polygon
def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
    """
        Plot all groups
        """
    from matplotlib.colors import to_rgba_array
    from matplotlib.image import AxesImage
    if not isinstance(coord, coord_cartesian):
        raise PlotnineError('geom_raster only works with cartesian coordinates')
    data = coord.transform(data, panel_params)
    x = data['x'].to_numpy().astype(float)
    y = data['y'].to_numpy().astype(float)
    facecolor = to_rgba_array(data['fill'].to_numpy())
    facecolor[:, 3] = data['alpha'].to_numpy()
    x_pos = ((x - x.min()) / resolution(x, False)).astype(int)
    y_pos = ((y - y.min()) / resolution(y, False)).astype(int)
    nrow = y_pos.max() + 1
    ncol = x_pos.max() + 1
    yidx, xidx = (nrow - y_pos - 1, x_pos)
    X = np.ones((nrow, ncol, 4))
    X[:, :, 3] = 0
    X[yidx, xidx] = facecolor
    im = AxesImage(ax, data=X, interpolation=params['interpolation'], origin='upper', extent=(data['xmin'].min(), data['xmax'].max(), data['ymin'].min(), data['ymax'].max()), rasterized=params['raster'], filterrad=params['filterrad'], zorder=params['zorder'])
    ax.add_image(im)