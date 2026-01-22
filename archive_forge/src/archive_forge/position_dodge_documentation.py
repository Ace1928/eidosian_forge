from __future__ import annotations
import typing
from contextlib import suppress
from copy import copy
import numpy as np
import pandas as pd
from .._utils import groupby_apply, match
from ..exceptions import PlotnineError
from .position import position

        Dodge overlapping interval

        Assumes that each set has the same horizontal position.
        