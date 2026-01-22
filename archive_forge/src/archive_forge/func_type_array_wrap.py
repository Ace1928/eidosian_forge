import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@type_callable('__array_wrap__')
def type_array_wrap(context):

    def typer(input_type, result):
        if isinstance(input_type, (IndexType, SeriesType)):
            return input_type.copy(dtype=result.dtype, ndim=result.ndim, layout=result.layout)
    return typer