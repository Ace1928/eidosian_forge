from __future__ import absolute_import, division, print_function
import itertools
import functools
import re
import types
from funcsigs.version import __version__
def formatannotation(annotation, base_module=None):
    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', '__builtin__', base_module):
            return annotation.__name__
        return annotation.__module__ + '.' + annotation.__name__
    return repr(annotation)