import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def insert_with_tags(self, iter, text, *tags):
    start_offset = iter.get_offset()
    self.insert(iter, text)
    if not tags:
        return
    start = self.get_iter_at_offset(start_offset)
    for tag in tags:
        self.apply_tag(tag, start, iter)