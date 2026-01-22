from .error import MarkedYAMLError
from .tokens import *
def scan_yaml_directive_number(self, start_mark):
    ch = self.peek()
    if not '0' <= ch <= '9':
        raise ScannerError('while scanning a directive', start_mark, 'expected a digit, but found %r' % ch, self.get_mark())
    length = 0
    while '0' <= self.peek(length) <= '9':
        length += 1
    value = int(self.prefix(length))
    self.forward(length)
    return value