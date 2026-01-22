import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def get_async_type(dep):
    async_or_dynamic = ''
    for key in dep.keys():
        if key in ['async', 'dynamic']:
            keyval = dep[key]
            if not isinstance(keyval, bool):
                keyval = "'{}'".format(keyval.lower())
            else:
                keyval = str(keyval).upper()
            async_or_dynamic = ', {} = {}'.format(key, keyval)
    return async_or_dynamic