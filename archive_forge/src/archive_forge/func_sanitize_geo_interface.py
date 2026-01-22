from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def sanitize_geo_interface(geo: MutableMapping) -> dict:
    """Santize a geo_interface to prepare it for serialization.

    * Make a copy
    * Convert type array or _Array to list
    * Convert tuples to lists (using json.loads/dumps)
    * Merge properties with geometry
    """
    geo = deepcopy(geo)
    for key in geo.keys():
        if str(type(geo[key]).__name__).startswith(('_Array', 'array')):
            geo[key] = geo[key].tolist()
    geo_dct: dict = json.loads(json.dumps(geo))
    if geo_dct['type'] == 'FeatureCollection':
        geo_dct = geo_dct['features']
        if len(geo_dct) > 0:
            for idx, feat in enumerate(geo_dct):
                geo_dct[idx] = merge_props_geom(feat)
    elif geo_dct['type'] == 'Feature':
        geo_dct = merge_props_geom(geo_dct)
    else:
        geo_dct = {'type': 'Feature', 'geometry': geo_dct}
    return geo_dct