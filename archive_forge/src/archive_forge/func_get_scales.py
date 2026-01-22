from __future__ import annotations
import typing
from contextlib import suppress
import numpy as np
from .._utils import match
from ..exceptions import PlotnineError
from ..iapi import labels_view, layout_details, pos_scales
def get_scales(self, i: int) -> pos_scales:
    """
        Return x & y scales for panel i

        Parameters
        ----------
        i : int
          Panel id

        Returns
        -------
        scales : types.SimpleNamespace
          Class attributes *x* for the x scale and *y*
          for the y scale of the panel

        """
    bool_idx = np.asarray(self.layout['PANEL']) == i
    idx = self.layout['SCALE_X'].loc[bool_idx].iloc[0]
    xsc = self.panel_scales_x[idx - 1]
    idx = self.layout['SCALE_Y'].loc[bool_idx].iloc[0]
    ysc = self.panel_scales_y[idx - 1]
    return pos_scales(x=xsc, y=ysc)