from __future__ import annotations
import datetime as dt
import os
import sys
from typing import Any, Iterable
import numpy as np
import param
def is_holoviews(obj: Any) -> bool:
    """
    Whether the object is a HoloViews type that can be rendered.
    """
    if 'holoviews' not in sys.modules:
        return False
    from holoviews.core.dimension import Dimensioned
    from holoviews.plotting import Plot
    return isinstance(obj, (Dimensioned, Plot))