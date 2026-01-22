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
@py2rpy.register(pandas.Categorical)
def py2rpy_categorical(obj):
    for c in obj.categories:
        if not isinstance(c, str):
            raise ValueError('Converting pandas "Category" series to R factor is only possible when categories are strings.')
    res = IntSexpVector(list((rinterface.NA_Integer if x == -1 else x + 1 for x in obj.codes)))
    res.do_slot_assign('levels', StrSexpVector(obj.categories))
    if obj.ordered:
        res.rclass = StrSexpVector(('ordered', 'factor'))
    else:
        res.rclass = StrSexpVector(('factor',))
    return res