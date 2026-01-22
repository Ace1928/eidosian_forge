import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def auto_set_column_width(self, col):
    """
        Automatically set the widths of given columns to optimal sizes.

        Parameters
        ----------
        col : int or sequence of ints
            The indices of the columns to auto-scale.
        """
    col1d = np.atleast_1d(col)
    if not np.issubdtype(col1d.dtype, np.integer):
        _api.warn_deprecated('3.8', name='col', message='%(name)r must be an int or sequence of ints. Passing other types is deprecated since %(since)s and will be removed %(removal)s.')
        return
    for cell in col1d:
        self._autoColumns.append(cell)
    self.stale = True