from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
def encode_nc3_variable(var):
    for coder in [coding.strings.EncodedStringCoder(allows_unicode=False), coding.strings.CharacterArrayCoder()]:
        var = coder.encode(var)
    data = _maybe_prepare_times(var)
    data = coerce_nc3_dtype(data)
    attrs = encode_nc3_attrs(var.attrs)
    return Variable(var.dims, data, attrs, var.encoding)