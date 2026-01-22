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
def print_largest(self, w=0, cutoff=0, **print3options):
    """Print the largest objects.

        The available options and defaults are:

         *w=0*           -- indentation for each line

         *cutoff=100*    -- number of largest objects to print

         *print3options* -- some keyword arguments, like Python 3+ print
        """
    c = int(cutoff) if cutoff else self._cutoff_
    n = min(len(self._ranks), max(c, 0))
    s = self._above_
    if n > 0 and s > 0:
        self._printf('%s%*d largest object%s (of %d over %d bytes%s)', linesep, w, n, _plural(n), self._ranked, s, _SI(s), **print3options)
        id2x = dict(((r.id, i) for i, r in enumerate(self._ranks)))
        for r in self._ranks[:n]:
            s, t = (r.size, r.format(self._clip_, id2x))
            self._printf('%*d bytes%s: %s', w, s, _SI(s), t, **print3options)