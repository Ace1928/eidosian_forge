import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def rawmethod(self, encoding):
    """Decorator for instance methods without any fancy shenanigans.
        The function must have the signature f(self, cmd, *args)
        where both self and cmd are just pointers to objc objects."""
    encoding = ensure_bytes(encoding)
    typecodes = parse_type_encoding(encoding)
    typecodes.insert(1, b'@:')
    encoding = b''.join(typecodes)

    def decorator(f):
        name = f.__name__.replace('_', ':')
        self.add_method(f, name, encoding)
        return f
    return decorator