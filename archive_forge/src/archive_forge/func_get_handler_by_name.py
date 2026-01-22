from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def get_handler_by_name(self, name):
    """Get a handler by its name."""
    return self._handlers.get(name)