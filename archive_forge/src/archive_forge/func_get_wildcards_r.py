import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def get_wildcards_r(prop_keys):
    wildcards = ''
    wildcards += ', '.join(("'{}'".format(p) for p in prop_keys if p.endswith('-*')))
    if wildcards == '':
        wildcards = 'NULL'
    return wildcards