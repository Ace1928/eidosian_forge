from __future__ import division, absolute_import, print_function
from itertools import chain, starmap
import itertools       # since zip_longest doesn't exist on Py2
from past.types import basestring
from past.utils import PY3
def oldzip(*args, **kwargs):
    return list(builtins.zip(*args, **kwargs))