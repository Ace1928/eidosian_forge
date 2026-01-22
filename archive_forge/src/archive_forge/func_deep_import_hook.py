imported from that module, which is useful when you're changing files deep
import builtins as builtin_mod
from contextlib import contextmanager
import importlib
import sys
from types import ModuleType
from warnings import warn
import types
def deep_import_hook(name, globals=None, locals=None, fromlist=None, level=-1):
    """Replacement for __import__()"""
    parent, buf = get_parent(globals, level)
    head, name, buf = load_next(parent, None if level < 0 else parent, name, buf)
    tail = head
    while name:
        tail, name, buf = load_next(tail, tail, name, buf)
    if tail is None:
        raise ValueError('Empty module name')
    if not fromlist:
        return head
    ensure_fromlist(tail, fromlist, buf, 0)
    return tail