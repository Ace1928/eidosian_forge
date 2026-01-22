import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_reg(self, reg):
    """match the given regular expression object to the current text
        position.

        if a match occurs, update the current text and line position.

        """
    mp = self.match_position
    match = reg.match(self.text, self.match_position)
    if match:
        start, end = match.span()
        self.match_position = end + 1 if end == start else end
        self.matched_lineno = self.lineno
        cp = mp - 1
        if cp >= 0 and cp < self.textlength:
            cp = self.text[:cp + 1].rfind('\n')
        self.matched_charpos = mp - cp
        self.lineno += self.text[mp:self.match_position].count('\n')
    return match