import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def connect_signals(self, obj_or_map):
    """Connect signals specified by this builder to a name, handler mapping.

            Connect signal, name, and handler sets specified in the builder with
            the given mapping "obj_or_map". The handler/value aspect of the mapping
            can also contain a tuple in the form of (handler [,arg1 [,argN]])
            allowing for extra arguments to be passed to the handler. For example:

            .. code-block:: python

                builder.connect_signals({'on_clicked': (on_clicked, arg1, arg2)})
            """
    self.connect_signals_full(_builder_connect_callback, obj_or_map)