import unicodedata
from wcwidth import wcwidth
from IPython.core.completer import (
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
import pygments.lexers as pygments_lexers
import os
import sys
import traceback
@property
def ipy_completer(self):
    if self._ipy_completer:
        return self._ipy_completer
    else:
        return self.shell.Completer