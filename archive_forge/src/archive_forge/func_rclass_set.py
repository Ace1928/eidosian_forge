import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
def rclass_set(scaps: _rinterface.CapsuleBase, value: 'typing.Union[StrSexpVector, str]') -> None:
    """ Set the R class.

    :param:`scaps` A capsule with a pointer to an R object.
    :param:`value` An R vector of strings."""
    if isinstance(value, StrSexpVector):
        value_r = value
    elif isinstance(value, str):
        value_r = StrSexpVector.from_iterable([value])
    else:
        raise TypeError('Value should a str or a rpy2.rinterface.sexp.StrSexpVector.')
    openrlib.rlib.Rf_setAttrib(scaps._cdata, openrlib.rlib.R_ClassSymbol, value_r.__sexp__._cdata)