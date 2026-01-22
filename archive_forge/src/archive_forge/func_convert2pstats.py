import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def convert2pstats(stats):
    from collections import defaultdict
    '\n    Converts the internal stat type of yappi(which is returned by a call to YFuncStats.get())\n    as pstats object.\n    '
    if not isinstance(stats, YFuncStats):
        raise YappiError('Source stats must be derived from YFuncStats.')
    import pstats

    class _PStatHolder:

        def __init__(self, d):
            self.stats = d

        def create_stats(self):
            pass

    def pstat_id(fs):
        return (fs.module, fs.lineno, fs.name)
    _pdict = {}
    _callers = defaultdict(dict)
    for fs in stats:
        for ct in fs.children:
            _callers[ct][pstat_id(fs)] = (ct.ncall, ct.nactualcall, ct.tsub, ct.ttot)
    for fs in stats:
        _pdict[pstat_id(fs)] = (fs.ncall, fs.nactualcall, fs.tsub, fs.ttot, _callers[fs])
    return pstats.Stats(_PStatHolder(_pdict))