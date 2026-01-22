from ..doctools import document
from .geom_point import geom_point

    Plot overlapping points

    This is a variant [](`~plotnine.geoms.geom_point`) that counts the number
    of observations at each location, then maps the count to point
    area. It useful when you have discrete data and overplotting.

    {usage}

    Parameters
    ----------
    {common_parameters}
    