import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def exclude_refs(self, *objs):
    """Exclude any references to the specified objects from sizing.

        While any references to the given objects are excluded, the
        objects will be sized if specified as positional arguments
        in subsequent calls to methods **asizeof** and **asizesof**.
        """
    for o in objs:
        self._seen.setdefault(id(o), 0)