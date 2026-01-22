from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def register_transformer(self, transformer):
    """Register a transformer instance."""
    if transformer not in self._transformers:
        self._transformers.append(transformer)
        self.sort_transformers()