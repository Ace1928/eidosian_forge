from __future__ import annotations
import typing
import numpy as np
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from ..scales.scale_shape import FILLED_SHAPES
from .geom import geom

        Draw a point in the box

        Parameters
        ----------
        data : Series
            Data Row
        da : DrawingArea
            Canvas
        lyr : layer
            Layer

        Returns
        -------
        out : DrawingArea
        