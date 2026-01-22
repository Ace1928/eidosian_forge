from __future__ import annotations
import re
import typing
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import fields
from typing import Any, Dict
import pandas as pd
from ..iapi import labels_view
from .evaluation import after_stat, stage
def rename_aesthetics(obj: THasAesNames) -> THasAesNames:
    """
    Rename aesthetics in obj

    Parameters
    ----------
    obj :
        Object that contains aesthetics names

    Returns
    -------
    :
        Object that contains aesthetics names
    """
    if isinstance(obj, dict):
        for name in tuple(obj.keys()):
            new_name = name.replace('colour', 'color')
            if name != new_name:
                obj[new_name] = obj.pop(name)
    elif isinstance(obj, Sequence):
        return type(obj)((s.replace('colour', 'color') for s in obj))
    elif obj.color is None and obj.colour is not None:
        obj.color, obj.colour = (obj.colour, None)
    return obj