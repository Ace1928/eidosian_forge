import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO, BinaryIO, Union
@staticmethod
def pp_format(printer, obj, stream, indent, allowance, context, level):
    if not obj.args and obj.state is None:
        stream.write(repr(obj))
        return
    if obj.state is None:
        stream.write(f'{obj.module}.{obj.name}')
        printer._format(obj.args, stream, indent + 1, allowance + 1, context, level)
        return
    if not obj.args:
        stream.write(f'{obj.module}.{obj.name}()(state=\n')
        indent += printer._indent_per_level
        stream.write(' ' * indent)
        printer._format(obj.state, stream, indent, allowance + 1, context, level + 1)
        stream.write(')')
        return
    raise Exception('Need to implement')