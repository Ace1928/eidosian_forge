from __future__ import annotations
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, order_as_data_mapping, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from ..mapping import aes
from .geom import geom
from .geom_segment import geom_segment

        Draw a vertical line in the box

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
        