import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@box(SeriesType)
def box_series(typ, val, c):
    """
    Convert a native series structure to a Series object.
    """
    series = make_series(c.context, c.builder, typ, value=val)
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(Series))
    indexobj = c.box(typ.index, series.index)
    arrayobj = c.box(typ.as_array, series.values)
    seriesobj = c.pyapi.call_function_objargs(classobj, (arrayobj, indexobj))
    return seriesobj