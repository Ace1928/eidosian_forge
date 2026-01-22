import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def deindent(self):
    allow_deindent = self._flags.indentation_level > 0 and (self._flags.parent is None or self._flags.indentation_level > self._flags.parent.indentation_level)
    if allow_deindent:
        self._flags.indentation_level -= 1
    self._output.set_indent(self._flags.indentation_level, self._flags.alignment)