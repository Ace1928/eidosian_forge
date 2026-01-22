import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def createbuilder_abi():
    ffibuilder = cffi.FFI()
    definitions = {}
    define_rlen_kind(ffibuilder, definitions)
    define_osname(definitions)
    r_h = read_source('R_API.h')
    if not os.name == 'nt':
        definitions['R_INTERFACE_PTRS'] = True
    cdef_r, _ = c_preprocess(iter(r_h.split('\n')), definitions=definitions, rownum=1)
    ffibuilder.set_source('_rinterface_cffi_abi', None)
    ffibuilder.cdef('\n'.join(cdef_r))
    return ffibuilder