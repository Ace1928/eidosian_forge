from __future__ import annotations
import datetime as dt
import os
import sys
from typing import Any, Iterable
import numpy as np
import param
def isIn(obj, objs):
    """
    Checks if the object is in the list of objects safely.
    """
    for o in objs:
        if o is obj:
            return True
        try:
            if o == obj:
                return True
        except Exception:
            pass
    return False