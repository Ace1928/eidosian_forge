import posixpath
import re
from ._hl.attrs import AttributeManager
from ._hl.base import HLObject
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import generics
def h5py_attr_completer(context, command):
    """Compute possible attr matches for nested dict-like objects"""
    base, attr = re_attr_match.split(command)[1:3]
    base = base.strip()
    try:
        obj = _retrieve_obj(base, context)
    except Exception:
        return []
    attrs = dir(obj)
    try:
        attrs = generics.complete_object(obj, attrs)
    except TryNext:
        pass
    try:
        omit__names = get_ipython().Completer.omit__names
    except AttributeError:
        omit__names = 0
    if omit__names == 1:
        attrs = [a for a in attrs if not a.startswith('__')]
    elif omit__names == 2:
        attrs = [a for a in attrs if not a.startswith('_')]
    return ['%s.%s' % (base, a) for a in attrs if a[:len(attr)] == attr]