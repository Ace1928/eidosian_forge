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
@deprecated('pyomo.common.register_executable(name) has been deprecated; explicit registration is no longer necessary', version='5.6.2')
def register_executable(name, validate=None):
    return Executable(name).rehash()