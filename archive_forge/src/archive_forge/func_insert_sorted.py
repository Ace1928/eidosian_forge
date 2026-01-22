import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
def insert_sorted(self, item, compare_func, *user_data):
    compare_func = wrap_list_store_sort_func(compare_func)
    return super(ListStore, self).insert_sorted(item, compare_func, *user_data)