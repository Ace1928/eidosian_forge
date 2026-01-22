from __future__ import annotations
import typing
from warnings import warn
import numpy as np
from .._utils import resolution
from ..coords import coord_cartesian
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from .geom import geom
from .geom_polygon import geom_polygon

        Plot all groups
        