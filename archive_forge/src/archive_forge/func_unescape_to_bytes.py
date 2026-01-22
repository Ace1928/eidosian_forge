import io
import sys
from typing import Any, List, Optional, Tuple
import dns.exception
import dns.name
import dns.ttl
def unescape_to_bytes(self) -> 'Token':
    unescaped = b''
    l = len(self.value)
    i = 0
    while i < l:
        c = self.value[i]
        i += 1
        if c == '\\':
            if i >= l:
                raise dns.exception.UnexpectedEnd
            c = self.value[i]
            i += 1
            if c.isdigit():
                if i >= l:
                    raise dns.exception.UnexpectedEnd
                c2 = self.value[i]
                i += 1
                if i >= l:
                    raise dns.exception.UnexpectedEnd
                c3 = self.value[i]
                i += 1
                if not (c2.isdigit() and c3.isdigit()):
                    raise dns.exception.SyntaxError
                codepoint = int(c) * 100 + int(c2) * 10 + int(c3)
                if codepoint > 255:
                    raise dns.exception.SyntaxError
                unescaped += b'%c' % codepoint
            else:
                unescaped += c.encode()
        else:
            unescaped += c.encode()
    return Token(self.ttype, bytes(unescaped))