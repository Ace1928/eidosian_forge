from __future__ import annotations
import typing
from copy import deepcopy
import pandas as pd
from .._utils import (
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError
from ..layer import layer
from ..mapping import aes
from abc import ABC
def finish_layer(self, data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """
        Modify data after the aesthetics have been mapped

        This can be used by stats that require access to the mapped
        values of the computed aesthetics, part 3 as shown below.

            1. stat computes and creates variables
            2. variables mapped to aesthetics
            3. stat sees and modifies data according to the
               aesthetic values

        The default to is to do nothing.

        Parameters
        ----------
        data :
            Data for the layer
        params :
            Paremeters

        Returns
        -------
        data :
            Modified data
        """
    return data