from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
@cached_property
def key_width(self):
    dim = self.is_vertical and 'width' or 'height'
    legend_key_dim = f'legend_key_{dim}'
    inherited = self.theme.T.get(legend_key_dim) is None
    scale = 1.45 if inherited else 1
    return np.round(self.theme.getp(legend_key_dim) * scale)