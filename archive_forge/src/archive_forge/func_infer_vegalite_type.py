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
def infer_vegalite_type(data: object) -> Union[InferredVegaLiteType, Tuple[InferredVegaLiteType, list]]:
    """
    From an array-like input, infer the correct vega typecode
    ('ordinal', 'nominal', 'quantitative', or 'temporal')

    Parameters
    ----------
    data: object
    """
    typ = infer_dtype(data, skipna=False)
    if typ in ['floating', 'mixed-integer-float', 'integer', 'mixed-integer', 'complex']:
        return 'quantitative'
    elif typ == 'categorical' and hasattr(data, 'cat') and data.cat.ordered:
        return ('ordinal', data.cat.categories.tolist())
    elif typ in ['string', 'bytes', 'categorical', 'boolean', 'mixed', 'unicode']:
        return 'nominal'
    elif typ in ['datetime', 'datetime64', 'timedelta', 'timedelta64', 'date', 'time', 'period']:
        return 'temporal'
    else:
        warnings.warn("I don't know how to infer vegalite type from '{}'.  Defaulting to nominal.".format(typ), stacklevel=1)
        return 'nominal'