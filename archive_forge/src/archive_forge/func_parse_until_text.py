import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def parse_until_text(self, watch_nesting, *text):
    startpos = self.match_position
    text_re = '|'.join(text)
    brace_level = 0
    paren_level = 0
    bracket_level = 0
    while True:
        match = self.match('#.*\\n')
        if match:
            continue
        match = self.match('(\\"\\"\\"|\\\'\\\'\\\'|\\"|\\\')[^\\\\]*?(\\\\.[^\\\\]*?)*\\1', re.S)
        if match:
            continue
        match = self.match('(%s)' % text_re)
        if match and (not (watch_nesting and (brace_level > 0 or paren_level > 0 or bracket_level > 0))):
            return (self.text[startpos:self.match_position - len(match.group(1))], match.group(1))
        elif not match:
            match = self.match('(.*?)(?=\\"|\\\'|#|%s)' % text_re, re.S)
        if match:
            brace_level += match.group(1).count('{')
            brace_level -= match.group(1).count('}')
            paren_level += match.group(1).count('(')
            paren_level -= match.group(1).count(')')
            bracket_level += match.group(1).count('[')
            bracket_level -= match.group(1).count(']')
            continue
        raise exceptions.SyntaxException('Expected: %s' % ','.join(text), **self.exception_kwargs)