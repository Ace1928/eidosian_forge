import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import na_values
from rpy2.rinterface import IntSexpVector
from rpy2.rinterface import ListSexpVector
from rpy2.rinterface import SexpVector
from rpy2.rinterface import StrSexpVector
import datetime
import functools
import math
import numpy  # type: ignore
import pandas  # type: ignore
import pandas.core.series  # type: ignore
from pandas.core.frame import DataFrame as PandasDataFrame  # type: ignore
from pandas.core.dtypes.api import is_datetime64_any_dtype  # type: ignore
import warnings
from collections import OrderedDict
from rpy2.robjects.vectors import (BoolVector,
import rpy2.robjects.numpy2ri as numpy2ri
@py2rpy.register(pandas.core.series.Series)
def py2rpy_pandasseries(obj):
    if obj.dtype.name == 'O':
        warnings.warn('Element "%s" is of dtype "O" and converted to R vector of strings.' % obj.name)
        res = StrVector(obj)
    elif obj.dtype.name == 'category':
        res = py2rpy_categorical(obj.cat)
        res = FactorVector(res)
    elif is_datetime64_any_dtype(obj.dtype):
        if obj.dt.tz:
            if obj.dt.tz is datetime.timezone.utc:
                tzname = 'UTC'
            else:
                tzname = obj.dt.tz.zone
        else:
            tzname = ''
        d = [IntVector([x.year for x in obj]), IntVector([x.month for x in obj]), IntVector([x.day for x in obj]), IntVector([x.hour for x in obj]), IntVector([x.minute for x in obj]), FloatSexpVector([x.second + x.microsecond * 1e-06 for x in obj])]
        res = ISOdatetime(*d, tz=StrSexpVector([tzname]))
        res = POSIXct(res)
    elif obj.dtype.type is str:
        res = _PANDASTYPE2RPY2[str](obj)
    elif obj.dtype.name in integer_array_types:
        res = _PANDASTYPE2RPY2[int](obj)
        if len(obj.shape) == 1:
            if obj.dtype != dt_O_type:
                res = as_vector(res)
    elif obj.dtype == dt_O_type:
        homogeneous_type = None
        for x in obj.values:
            if x is None:
                continue
            if homogeneous_type is None:
                homogeneous_type = type(x)
                continue
            if type(x) is not homogeneous_type and (not (isinstance(x, float) and math.isnan(x) or pandas.isna(x))):
                raise ValueError('Series can only be of one type, or None (and here we have %s and %s). If happening with a pandas DataFrame the method infer_objects() will normalize data types before conversion.' % (homogeneous_type, type(x)))
        res = _PANDASTYPE2RPY2[homogeneous_type](obj)
    elif type(obj.dtype) in (pandas.Float64Dtype, pandas.BooleanDtype):
        res = _PANDASTYPE2RPY2[type(obj.dtype)](obj)
    else:
        func = numpy2ri.converter.py2rpy.registry[numpy.ndarray]
        res = func(obj.values)
        if len(obj.shape) == 1:
            if obj.dtype != dt_O_type:
                res = as_vector(res)
    if obj.ndim == 1:
        res.do_slot_assign('names', StrVector(tuple((str(x) for x in obj.index))))
    else:
        res.do_slot_assign('dimnames', SexpVector(conversion.converter_ctx.get().py2rpy(obj.index)))
    return res