from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def transform_line(self, line, continue_prompt):
    """Calls the enabled transformers in order of increasing priority."""
    for transformer in self.transformers:
        if transformer.enabled:
            line = transformer.transform(line, continue_prompt)
    return line