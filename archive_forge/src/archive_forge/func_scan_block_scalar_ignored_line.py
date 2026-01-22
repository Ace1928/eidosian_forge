from .error import MarkedYAMLError
from .tokens import *
def scan_block_scalar_ignored_line(self, start_mark):
    while self.peek() == ' ':
        self.forward()
    if self.peek() == '#':
        while self.peek() not in '\x00\r\n\x85\u2028\u2029':
            self.forward()
    ch = self.peek()
    if ch not in '\x00\r\n\x85\u2028\u2029':
        raise ScannerError('while scanning a block scalar', start_mark, 'expected a comment or a line break, but found %r' % ch, self.get_mark())
    self.scan_line_break()