import sys
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from holoviews.core.util import isscalar, unique_iterator, unique_array
from holoviews.core.data import Dataset, Interface, MultiInterface, PandasAPI
from holoviews.core.data.interface import DataError
from holoviews.core.data import PandasInterface
from holoviews.core.data.spatialpandas import get_value_array
from holoviews.core.dimension import dimension_name
from holoviews.element import Path
from ..util import asarray, geom_to_array, geom_types, geom_length
from .geom_dict import geom_from_dict
def from_multi(eltype, data, kdims, vdims):
    """Converts list formats into geopandas.GeoDataFrame.

    Args:
        eltype: Element type to convert
        data: The original data
        kdims: The declared key dimensions
        vdims: The declared value dimensions

    Returns:
        A GeoDataFrame containing the data in the list based format.
    """
    from geopandas import GeoDataFrame
    new_data = []
    types = []
    xname, yname = (kd.name for kd in kdims[:2])
    for d in data:
        types.append(type(d))
        if isinstance(d, dict):
            d = {k: v if isscalar(v) else asarray(v) for k, v in d.items()}
            new_data.append(d)
            continue
        new_el = eltype(d, kdims, vdims)
        if new_el.interface is GeoPandasInterface:
            types[-1] = GeoDataFrame
            new_data.append(new_el.data)
            continue
        new_dict = {}
        for d in new_el.dimensions():
            if d in (xname, yname):
                scalar = False
            else:
                scalar = new_el.interface.isscalar(new_el, d)
            vals = new_el.dimension_values(d, not scalar)
            new_dict[d.name] = vals[0] if scalar else vals
        new_data.append(new_dict)
    if len(set(types)) > 1:
        raise DataError('Mixed types not supported')
    if new_data and types[0] is GeoDataFrame:
        data = pd.concat(new_data)
    else:
        columns = [d.name for d in kdims + vdims if d not in (xname, yname)]
        geom = GeoPandasInterface.geom_type(eltype)
        if not len(data):
            return GeoDataFrame([], columns=['geometry'] + columns)
        data = to_geopandas(new_data, xname, yname, columns, geom)
    return data