from .error import MarkedYAMLError
from .tokens import *
def scan_tag_directive_prefix(self, start_mark):
    value = self.scan_tag_uri('directive', start_mark)
    ch = self.peek()
    if ch not in '\x00 \r\n\x85\u2028\u2029':
        raise ScannerError('while scanning a directive', start_mark, "expected ' ', but found %r" % ch, self.get_mark())
    return value