from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from mizani.bounds import expand_range_distinct
from .._utils import match
from ..doctools import document
from ..iapi import range_view, scale_view
from ._expand import expand_range
from .range import RangeDiscrete
from .scale import scale

        Inverse Transform dataframe
        