import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def getimport(obj, alias='', verify=True, builtin=False, enclosing=False):
    """get the likely import string for the given object

    obj is the object to inspect
    If verify=True, then test the import string before returning it.
    If builtin=True, then force an import for builtins where possible.
    If enclosing=True, get the import for the outermost enclosing callable.
    If alias is provided, then rename the object on import.
    """
    if enclosing:
        from .detect import outermost
        _obj = outermost(obj)
        obj = _obj if _obj else obj
    qual = _namespace(obj)
    head = '.'.join(qual[:-1])
    tail = qual[-1]
    try:
        name = repr(obj).split('<', 1)[1].split('>', 1)[1]
        name = None
    except Exception:
        if head in ['builtins', '__builtin__']:
            name = repr(obj)
        else:
            name = repr(obj).split('(')[0]
    if name:
        try:
            return _getimport(head, name, alias, verify, builtin)
        except ImportError:
            pass
        except SyntaxError:
            if head in ['builtins', '__builtin__']:
                _alias = '%s = ' % alias if alias else ''
                if alias == name:
                    _alias = ''
                return _alias + '%s\n' % name
            else:
                pass
    try:
        return _getimport(head, tail, alias, verify, builtin)
    except ImportError:
        raise
    except SyntaxError:
        if head in ['builtins', '__builtin__']:
            _alias = '%s = ' % alias if alias else ''
            if alias == tail:
                _alias = ''
            return _alias + '%s\n' % tail
        raise