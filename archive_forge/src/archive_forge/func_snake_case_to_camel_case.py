import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def snake_case_to_camel_case(namestring):
    s = namestring.split('_')
    return s[0] + ''.join((w.capitalize() for w in s[1:]))