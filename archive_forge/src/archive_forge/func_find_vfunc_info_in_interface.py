import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def find_vfunc_info_in_interface(bases, vfunc_name):
    for base in bases:
        if base is GInterface or not issubclass(base, GInterface) or (not hasattr(base, '__info__')):
            continue
        if isinstance(base.__info__, InterfaceInfo):
            for vfunc in base.__info__.get_vfuncs():
                if vfunc.get_name() == vfunc_name:
                    return vfunc
        vfunc = find_vfunc_info_in_interface(base.__bases__, vfunc_name)
        if vfunc is not None:
            return vfunc
    return None