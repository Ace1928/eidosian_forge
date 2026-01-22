from contextlib import contextmanager
from .utils import hashable
from .dispatch import dispatch
@dispatch(object)
def isvar(o):
    return not not _glv and hashable(o) and (o in _glv)