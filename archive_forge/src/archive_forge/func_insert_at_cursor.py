import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def insert_at_cursor(self, text, length=-1):
    if not isinstance(text, str):
        raise TypeError('text must be a string, not %s' % type(text))
    Gtk.TextBuffer.insert_at_cursor(self, text, length)