import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def combo_row_separator_func(self, func, user_data=_unset):

    def callback(*args):
        if args[-1] == _unset:
            args = args[:-1]
        return func(*args)
    orig_combo_row_separator_func(self, callback, user_data)