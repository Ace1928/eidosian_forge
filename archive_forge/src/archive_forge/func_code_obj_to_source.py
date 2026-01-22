import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
def code_obj_to_source(co):
    """
    Converts a code object to source code to provide a suitable representation for the compiler when
    the actual source code is not found.

    This is a work in progress / proof of concept / not ready to be used.
    """
    ret = _PyCodeToSource(co).disassemble()
    if DEBUG:
        print(ret)
    return ret