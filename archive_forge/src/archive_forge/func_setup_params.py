from __future__ import annotations
import typing
from copy import copy
import numpy as np
from ..iapi import panel_ranges
def setup_params(self, data: list[pd.DataFrame]):
    """
        Create additional parameters

        A coordinate system may need to create parameters
        depending on the *original* data that the layers get.

        Parameters
        ----------
        data :
            Data for each layer before it is manipulated in
            any way.
        """
    self.params = {}