from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def get_handler_by_esc(self, esc_str):
    """Get a handler by its escape string."""
    return self._esc_handlers.get(esc_str)