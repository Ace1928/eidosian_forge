from __future__ import annotations
import typing
from .._utils import SIZE_FACTOR, to_rgba
from ..coords import coord_flip
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_path import geom_path
from .geom_polygon import geom_polygon

    Ribbon plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    outline_type : Literal["upper", "lower", "both", "full"], default="both"
        How to stroke to outline of the region / area.
        If `upper`, draw only upper bounding line.
        If `lower`, draw only lower bounding line.
        If `both`, draw both upper & lower bounding lines.
        If `full`, draw closed polygon around the area.
    