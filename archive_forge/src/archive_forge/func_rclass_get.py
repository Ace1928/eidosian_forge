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
def rclass_get(scaps: _rinterface.CapsuleBase) -> StrSexpVector:
    """ Get the R class name.

    If no specific attribute "class" is defined from the objects, this
    will perform the equivalent of R_data_class()
    (src/main/attrib.c in the R source code).
    """
    rlib = openrlib.rlib
    with memorymanagement.rmemory() as rmemory:
        classes = rmemory.protect(rlib.Rf_getAttrib(scaps._cdata, rlib.R_ClassSymbol))
        if rlib.Rf_length(classes) == 0:
            classname: typing.Tuple[str, ...]
            dim = rmemory.protect(rlib.Rf_getAttrib(scaps._cdata, rlib.R_DimSymbol))
            ndim = rlib.Rf_length(dim)
            if ndim > 0:
                if ndim == 2:
                    if int(RVersion()['major']) >= 4:
                        classname = ('matrix', 'array')
                    else:
                        classname = ('matrix',)
                else:
                    classname = ('array',)
            else:
                typeof = RTYPES(scaps.typeof)
                if typeof in (RTYPES.CLOSXP, RTYPES.SPECIALSXP, RTYPES.BUILTINSXP):
                    classname = ('function',)
                elif typeof == RTYPES.REALSXP:
                    classname = ('numeric',)
                elif typeof == RTYPES.SYMSXP:
                    classname = ('name',)
                elif typeof == RTYPES.LANGSXP:
                    symb = rlib.CAR(scaps._cdata)
                    if openrlib.rlib.Rf_isSymbol(symb):
                        symb_rstr = openrlib.rlib.PRINTNAME(symb)
                        symb_str = conversion._cchar_to_str(openrlib.rlib.R_CHAR(symb_rstr), conversion._R_ENC_PY[openrlib.rlib.Rf_getCharCE(symb_rstr)])
                        if symb_str in ('if', 'while', 'for', '=', '<-', '(', '{'):
                            classname = (symb_str,)
                        else:
                            classname = ('call',)
                    else:
                        classname = ('call',)
                else:
                    classname = (_TYPE2STR.get(typeof, str(typeof)),)
            classes = StrSexpVector.from_iterable(classname)
        else:
            classes = conversion._cdata_to_rinterface(classes)
    return classes