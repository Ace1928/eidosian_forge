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
def geo_column(cls, dataset):
    return 'geometry'