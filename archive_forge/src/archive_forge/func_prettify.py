import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def prettify(obj):
    if hasattr(obj, 'getSource'):
        return obj.getSource()
    else:
        t = type(obj)
        if t in _SIMPLE_BUILTINS:
            return repr(obj)
        elif t is dict:
            out = ['{']
            for k, v in obj.items():
                out.append(f'\n\x00{prettify(k)}: {prettify(v)},')
            out.append(len(obj) and '\n\x00}' or '}')
            return ''.join(out)
        elif t is list:
            out = ['[']
            for x in obj:
                out.append('\n\x00%s,' % prettify(x))
            out.append(len(obj) and '\n\x00]' or ']')
            return ''.join(out)
        elif t is tuple:
            out = ['(']
            for x in obj:
                out.append('\n\x00%s,' % prettify(x))
            out.append(len(obj) and '\n\x00)' or ')')
            return ''.join(out)
        else:
            raise TypeError(f'Unsupported type {t} when trying to prettify {obj}.')