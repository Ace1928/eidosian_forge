from __future__ import annotations
import typing
from .._utils import to_rgba
from ..doctools import document
from .geom import geom
from .geom_line import geom_line, geom_path
from .geom_ribbon import geom_ribbon

        Draw letter 'a' in the box

        Parameters
        ----------
        data : dataframe
            Legend data
        da : DrawingArea
            Canvas
        lyr : layer
            Layer

        Returns
        -------
        out : DrawingArea
        