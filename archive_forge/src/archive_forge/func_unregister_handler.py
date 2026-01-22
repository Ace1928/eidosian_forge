from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def unregister_handler(self, name, handler, esc_strings):
    """Unregister a handler instance by name with esc_strings."""
    try:
        del self._handlers[name]
    except KeyError:
        pass
    for esc_str in esc_strings:
        h = self._esc_handlers.get(esc_str)
        if h is handler:
            del self._esc_handlers[esc_str]