from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def install_aliases():
    """
    Monkey-patches the standard library in Py2.6/7 to provide
    aliases for better Py3 compatibility.
    """
    if PY3:
        return
    for newmodname, newobjname, oldmodname, oldobjname in MOVES:
        __import__(newmodname)
        newmod = sys.modules[newmodname]
        __import__(oldmodname)
        oldmod = sys.modules[oldmodname]
        obj = getattr(oldmod, oldobjname)
        setattr(newmod, newobjname, obj)
    import urllib
    from future.backports.urllib import request
    from future.backports.urllib import response
    from future.backports.urllib import parse
    from future.backports.urllib import error
    from future.backports.urllib import robotparser
    urllib.request = request
    urllib.response = response
    urllib.parse = parse
    urllib.error = error
    urllib.robotparser = robotparser
    sys.modules['urllib.request'] = request
    sys.modules['urllib.response'] = response
    sys.modules['urllib.parse'] = parse
    sys.modules['urllib.error'] = error
    sys.modules['urllib.robotparser'] = robotparser
    try:
        import test
    except ImportError:
        pass
    try:
        from future.moves.test import support
    except ImportError:
        pass
    else:
        test.support = support
        sys.modules['test.support'] = support
    try:
        import dbm
    except ImportError:
        pass
    else:
        from future.moves.dbm import dumb
        dbm.dumb = dumb
        sys.modules['dbm.dumb'] = dumb
        try:
            from future.moves.dbm import gnu
        except ImportError:
            pass
        else:
            dbm.gnu = gnu
            sys.modules['dbm.gnu'] = gnu
        try:
            from future.moves.dbm import ndbm
        except ImportError:
            pass
        else:
            dbm.ndbm = ndbm
            sys.modules['dbm.ndbm'] = ndbm