import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(unmatched_quote)
def t_UNMATCHED_QUOTE(self, t):
    msg = "Unmatched '"
    self._error(msg, t)