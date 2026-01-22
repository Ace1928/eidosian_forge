import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def scroll_to_cell(self, path, column=None, use_align=False, row_align=0.0, col_align=0.0):
    if not isinstance(path, Gtk.TreePath):
        path = TreePath(path)
    super(TreeView, self).scroll_to_cell(path, column, use_align, row_align, col_align)