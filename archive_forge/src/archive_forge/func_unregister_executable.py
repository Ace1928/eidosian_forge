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
@deprecated('pyomo.common.unregister_executable(name) has been deprecated; use Executable(name).disable()', version='5.6.2')
def unregister_executable(name):
    Executable(name).disable()