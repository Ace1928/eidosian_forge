from __future__ import annotations
import typing
from abc import ABC
from copy import copy
from warnings import warn
import numpy as np
from .._utils import check_required_aesthetics, groupby_apply
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import X_AESTHETICS, Y_AESTHETICS
@classmethod
def collide2(cls, data, params):
    """
        Calculate boundaries of geometry object

        Uses Strategy
        """
    data, width = cls._collide_setup(data, params)
    if params.get('width', None) is None:
        params['width'] = width
    if params and 'reverse' in params and params['reverse']:
        data['-group'] = -data['group']
        idx = data.sort_values(['x', '-group'], kind='mergesort').index
        del data['-group']
    else:
        idx = data.sort_values(['x', 'group'], kind='mergesort').index
    data = data.loc[idx, :]
    data.reset_index(inplace=True, drop=True)
    return cls.strategy(data, params)