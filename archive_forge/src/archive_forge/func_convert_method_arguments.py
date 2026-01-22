import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def convert_method_arguments(encoding, args):
    """Used by ObjCSubclass to convert Objective-C method arguments to
    Python values before passing them on to the Python-defined method."""
    new_args = []
    arg_encodings = parse_type_encoding(encoding)[3:]
    for e, a in zip(arg_encodings, args):
        if e == b'@':
            new_args.append(ObjCInstance(a))
        elif e == b'#':
            new_args.append(ObjCClass(a))
        else:
            new_args.append(a)
    return new_args