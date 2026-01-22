from __future__ import annotations
from collections.abc import Sequence
import io
from typing import TYPE_CHECKING, Any, cast
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
from contourpy import FillType, LineType
from contourpy.convert import convert_filled, convert_lines
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.mpl_util import filled_to_mpl_paths, lines_to_mpl_paths
from contourpy.util.renderer import Renderer
def z_levels(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, lower_level: float, upper_level: float | None=None, ax: Axes | int=0, color: str='green') -> None:
    ax = self._get_ax(ax)
    x, y = self._grid_as_2d(x, y)
    z = np.asarray(z)
    ny, nx = z.shape
    for j in range(ny):
        for i in range(nx):
            zz = z[j, i]
            if upper_level is not None and zz > upper_level:
                z_level = 2
            elif zz > lower_level:
                z_level = 1
            else:
                z_level = 0
            ax.text(x[j, i], y[j, i], str(z_level), ha='left', va='bottom', color=color, clip_on=True)