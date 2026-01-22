from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
def is_builtin_module(module_name):
    """Test if a module is a builtin module (numpy, math, ...)."""
    module_name = module_name.split('.')[0]
    return module_name in MODULES