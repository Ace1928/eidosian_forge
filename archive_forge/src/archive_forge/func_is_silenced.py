from __future__ import annotations
import logging # isort:skip
import contextlib
from typing import (
from ...model import Model
from ...settings import settings
from ...util.dataclasses import dataclass, field
from .issue import Warning
def is_silenced(warning: Warning) -> bool:
    """ Check if a warning has been silenced.

    Args:
        warning (Warning) : Bokeh warning to check

    Returns:
        bool

    """
    return warning in __silencers__