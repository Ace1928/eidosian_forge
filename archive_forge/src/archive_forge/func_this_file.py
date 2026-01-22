import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
def this_file(stack_offset=1):
    """Returns the file name for the module that calls this function.

    This function is more reliable than __file__ on platforms like
    Windows and in situations where the program has called
    os.chdir().

    """
    callerFrame = inspect.currentframe()
    while stack_offset:
        callerFrame = callerFrame.f_back
        stack_offset -= 1
    frameName = callerFrame.f_code.co_filename
    if frameName and frameName[0] == '<' and (frameName[-1] == '>'):
        return frameName
    return os.path.abspath(inspect.getfile(callerFrame))