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
def create_geoms(self):
    """
        Return self if colorbar will be drawn and None if not

        This guide is not geom based
        """
    for l in self.plot_layers:
        exclude = set()
        if isinstance(l.show_legend, dict):
            l.show_legend = rename_aesthetics(l.show_legend)
            exclude = {ae for ae, val in l.show_legend.items() if not val}
        elif l.show_legend not in (None, True):
            continue
        matched = self.legend_aesthetics(l)
        if set(matched) - exclude:
            break
    else:
        return None
    return self