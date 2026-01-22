import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_python_block(self):
    match = self.match('<%(!)?')
    if match:
        line, pos = (self.matched_lineno, self.matched_charpos)
        text, end = self.parse_until_text(False, '%>')
        text = adjust_whitespace(text) + '\n'
        self.append_node(parsetree.Code, text, match.group(1) == '!', lineno=line, pos=pos)
        return True
    else:
        return False