import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def make_c_source(ffi, module_name, preamble, target_c_file, verbose=False):
    assert preamble is not None
    return _make_c_or_py_source(ffi, module_name, preamble, target_c_file, verbose)