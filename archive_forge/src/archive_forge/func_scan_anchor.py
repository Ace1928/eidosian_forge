from .error import MarkedYAMLError
from .tokens import *
def scan_anchor(self, TokenClass):
    start_mark = self.get_mark()
    indicator = self.peek()
    if indicator == '*':
        name = 'alias'
    else:
        name = 'anchor'
    self.forward()
    length = 0
    ch = self.peek(length)
    while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or (ch in '-_'):
        length += 1
        ch = self.peek(length)
    if not length:
        raise ScannerError('while scanning an %s' % name, start_mark, 'expected alphabetic or numeric character, but found %r' % ch, self.get_mark())
    value = self.prefix(length)
    self.forward(length)
    ch = self.peek()
    if ch not in '\x00 \t\r\n\x85\u2028\u2029?:,]}%@`':
        raise ScannerError('while scanning an %s' % name, start_mark, 'expected alphabetic or numeric character, but found %r' % ch, self.get_mark())
    end_mark = self.get_mark()
    return TokenClass(value, start_mark, end_mark)