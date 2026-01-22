import copy
import os
import shutil
import warnings
import sys
import importlib
import uuid
import hashlib
from ._all_keywords import julia_keywords
from ._py_components_generation import reorder_props
def jl_package_name(namestring):
    s = namestring.split('_')
    return ''.join((w.capitalize() for w in s))