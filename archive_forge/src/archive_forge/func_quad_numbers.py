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
def quad_numbers(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, ax: Axes | int=0, color: str='blue') -> None:
    ax = self._get_ax(ax)
    x, y = self._grid_as_2d(x, y)
    z = np.asarray(z)
    ny, nx = z.shape
    for j in range(1, ny):
        for i in range(1, nx):
            quad = i + j * nx
            xmid = x[j - 1:j + 1, i - 1:i + 1].mean()
            ymid = y[j - 1:j + 1, i - 1:i + 1].mean()
            ax.text(xmid, ymid, str(quad), ha='center', va='center', color=color, clip_on=True)