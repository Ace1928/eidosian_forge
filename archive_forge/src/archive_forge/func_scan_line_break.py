from .error import MarkedYAMLError
from .tokens import *
def scan_line_break(self):
    ch = self.peek()
    if ch in '\r\n\x85':
        if self.prefix(2) == '\r\n':
            self.forward(2)
        else:
            self.forward()
        return '\n'
    elif ch in '\u2028\u2029':
        self.forward()
        return ch
    return ''