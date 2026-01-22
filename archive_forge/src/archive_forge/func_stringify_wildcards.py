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
def stringify_wildcards(wclist, no_symbol=False):
    if no_symbol:
        wcstring = '|'.join(('{}-'.format(item) for item in wclist))
    else:
        wcstring = ', '.join(('Symbol("{}-")'.format(item) for item in wclist))
    return wcstring