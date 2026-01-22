from .error import MarkedYAMLError
from .tokens import *
def stale_possible_simple_keys(self):
    for level in list(self.possible_simple_keys):
        key = self.possible_simple_keys[level]
        if key.line != self.line or self.index - key.index > 1024:
            if key.required:
                raise ScannerError('while scanning a simple key', key.mark, "could not find expected ':'", self.get_mark())
            del self.possible_simple_keys[level]