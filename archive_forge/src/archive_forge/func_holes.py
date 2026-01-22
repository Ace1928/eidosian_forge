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
def holes(cls, dataset):
    from shapely.geometry import Polygon, MultiPolygon
    geom = dataset.data['geometry']
    if isinstance(geom, Polygon):
        return [[[geom_to_array(h) for h in geom.interiors]]]
    elif isinstance(geom, MultiPolygon):
        return [[[geom_to_array(h) for h in g.interiors] for g in geom.geoms]]
    return []