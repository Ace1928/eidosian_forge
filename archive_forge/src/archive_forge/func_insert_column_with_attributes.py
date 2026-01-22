import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def insert_column_with_attributes(self, position, title, cell, **kwargs):
    column = TreeViewColumn()
    column.set_title(title)
    column.pack_start(cell, False)
    self.insert_column(column, position)
    column.set_attributes(cell, **kwargs)