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
def print_typedefs(self, w=0, **print3options):
    """Print the types and dict tables.

        The available options and defaults are:

             *w=0*           -- indentation for each line

             *print3options* -- some keyword arguments, like Python 3+ print
        """
    for k in _all_kinds:
        t = [(self._prepr(a), v) for a, v in _items(_typedefs) if v.kind == k and (v.both or self._code_)]
        if t:
            self._printf('%s%*d %s type%s:  basicsize, itemsize, _len_(), _refs()', linesep, w, len(t), k, _plural(len(t)), **print3options)
            for a, v in sorted(t):
                self._printf('%*s %s:  %s', w, _NN, a, v, **print3options)
    t = sum((len(v) for v in _values(_dict_types)))
    if t:
        self._printf('%s%*d dict/-like classes:', linesep, w, t, **print3options)
        for m, v in _items(_dict_types):
            self._printf('%*s %s:  %s', w, _NN, m, self._prepr(v), **print3options)