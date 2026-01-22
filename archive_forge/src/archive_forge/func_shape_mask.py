import sys
from collections import OrderedDict
import numpy as np
from holoviews.core.data import Interface, DictInterface, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.core.data.spatialpandas import to_geom_dict
from holoviews.core.dimension import dimension_name
from holoviews.core.util import isscalar
from ..util import asarray, geom_types, geom_to_array, geom_length
@classmethod
def shape_mask(cls, dataset, selection):
    xdim, ydim = cls.geom_dims(dataset)
    xsel = selection.pop(xdim.name, None)
    ysel = selection.pop(ydim.name, None)
    if xsel is None and ysel is None:
        return dataset.data
    from shapely.geometry import box
    if xsel is None:
        x0, x1 = cls.range(dataset, xdim)
    elif isinstance(xsel, slice):
        x0, x1 = (xsel.start, xsel.stop)
    elif isinstance(xsel, tuple):
        x0, x1 = xsel
    else:
        raise ValueError(f'Only slicing is supported on geometries, {xdim} selection is of type {type(xsel).__name__}.')
    if ysel is None:
        y0, y1 = cls.range(dataset, ydim)
    elif isinstance(ysel, slice):
        y0, y1 = (ysel.start, ysel.stop)
    elif isinstance(ysel, tuple):
        y0, y1 = ysel
    else:
        raise ValueError(f'Only slicing is supported on geometries, {ydim} selection is of type {type(ysel).__name__}.')
    bounds = box(x0, y0, x1, y1)
    geom = dataset.data['geometry']
    geom = geom.intersection(bounds)
    new_data = dict(dataset.data, geometry=geom)
    return new_data