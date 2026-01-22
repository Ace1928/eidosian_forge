import re
from .ply import lex
from .ply.lex import TOKEN
def t_ppline_NEWLINE(self, t):
    """\\n"""
    if self.pp_line is None:
        self._error('line number missing in #line', t)
    else:
        self.lexer.lineno = int(self.pp_line)
        if self.pp_filename is not None:
            self.filename = self.pp_filename
    t.lexer.begin('INITIAL')