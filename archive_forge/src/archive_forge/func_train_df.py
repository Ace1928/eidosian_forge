from __future__ import annotations
import typing
from abc import ABC
from copy import copy, deepcopy
from warnings import warn
import numpy as np
from mizani.palettes import identity_pal
from .._utils.registry import Register
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import is_position_aes, rename_aesthetics
from .range import Range
def train_df(self, df: pd.DataFrame):
    """
        Train scale from a dataframe
        """
    aesthetics = sorted(set(self.aesthetics) & set(df.columns))
    for ae in aesthetics:
        self.train(df[ae])