from .error import MarkedYAMLError
from .tokens import *
def scan_to_next_token(self):
    if self.index == 0 and self.peek() == '\ufeff':
        self.forward()
    found = False
    while not found:
        while self.peek() == ' ':
            self.forward()
        if self.peek() == '#':
            while self.peek() not in '\x00\r\n\x85\u2028\u2029':
                self.forward()
        if self.scan_line_break():
            if not self.flow_level:
                self.allow_simple_key = True
        else:
            found = True