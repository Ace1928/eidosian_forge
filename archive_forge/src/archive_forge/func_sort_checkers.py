from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def sort_checkers(self):
    """Sort the checkers by priority.

        This must be called after the priority of a checker is changed.
        The :meth:`register_checker` method calls this automatically.
        """
    self._checkers.sort(key=lambda x: x.priority)