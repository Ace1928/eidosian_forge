from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def unregister_checker(self, checker):
    """Unregister a checker instance."""
    if checker in self._checkers:
        self._checkers.remove(checker)