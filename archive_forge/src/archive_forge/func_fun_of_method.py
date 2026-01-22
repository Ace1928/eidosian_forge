import operator
import sys
from functools import reduce
from importlib import import_module
from types import ModuleType
def fun_of_method(method):
    return method