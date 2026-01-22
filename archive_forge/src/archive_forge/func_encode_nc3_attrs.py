from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
def encode_nc3_attrs(attrs):
    return {k: encode_nc3_attr_value(v) for k, v in attrs.items()}