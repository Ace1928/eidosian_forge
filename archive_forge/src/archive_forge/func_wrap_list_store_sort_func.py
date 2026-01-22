import functools
import types
import warnings
import importlib
import sys
from gi import PyGIDeprecationWarning
from gi._gi import CallableInfo, pygobject_new_full
from gi._constants import \
from pkgutil import extend_path
def wrap_list_store_sort_func(func):

    def wrap(a, b, *user_data):
        a = pygobject_new_full(a, False)
        b = pygobject_new_full(b, False)
        return func(a, b, *user_data)
    return wrap