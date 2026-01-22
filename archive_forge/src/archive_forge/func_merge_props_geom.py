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
def merge_props_geom(feat: dict) -> dict:
    """
    Merge properties with geometry
    * Overwrites 'type' and 'geometry' entries if existing
    """
    geom = {k: feat[k] for k in ('type', 'geometry')}
    try:
        feat['properties'].update(geom)
        props_geom = feat['properties']
    except (AttributeError, KeyError):
        props_geom = geom
    return props_geom