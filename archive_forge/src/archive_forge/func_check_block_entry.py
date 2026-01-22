from .error import MarkedYAMLError
from .tokens import *
def check_block_entry(self):
    return self.peek(1) in '\x00 \t\r\n\x85\u2028\u2029'