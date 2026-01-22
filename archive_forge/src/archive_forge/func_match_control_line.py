import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_control_line(self):
    match = self.match('(?<=^)[\\t ]*(%(?!%)|##)[\\t ]*((?:(?:\\\\\\r?\\n)|[^\\r\\n])*)(?:\\r?\\n|\\Z)', re.M)
    if not match:
        return False
    operator = match.group(1)
    text = match.group(2)
    if operator == '%':
        m2 = re.match('(end)?(\\w+)\\s*(.*)', text)
        if not m2:
            raise exceptions.SyntaxException("Invalid control line: '%s'" % text, **self.exception_kwargs)
        isend, keyword = m2.group(1, 2)
        isend = isend is not None
        if isend:
            if not len(self.control_line):
                raise exceptions.SyntaxException("No starting keyword '%s' for '%s'" % (keyword, text), **self.exception_kwargs)
            elif self.control_line[-1].keyword != keyword:
                raise exceptions.SyntaxException("Keyword '%s' doesn't match keyword '%s'" % (text, self.control_line[-1].keyword), **self.exception_kwargs)
        self.append_node(parsetree.ControlLine, keyword, isend, text)
    else:
        self.append_node(parsetree.Comment, text)
    return True