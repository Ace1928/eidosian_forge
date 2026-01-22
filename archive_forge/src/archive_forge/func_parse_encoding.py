from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
def parse_encoding(fp):
    """Deduce the encoding of a Python source file (binary mode) from magic
    comment.

    It does this in the same way as the `Python interpreter`__

    .. __: http://docs.python.org/ref/encodings.html

    The ``fp`` argument should be a seekable file object in binary mode.
    """
    pos = fp.tell()
    fp.seek(0)
    try:
        line1 = fp.readline()
        has_bom = line1.startswith(codecs.BOM_UTF8)
        if has_bom:
            line1 = line1[len(codecs.BOM_UTF8):]
        m = _PYTHON_MAGIC_COMMENT_re.match(line1.decode('ascii', 'ignore'))
        if not m:
            try:
                parse(line1.decode('ascii', 'ignore'))
            except (ImportError, SyntaxError):
                pass
            else:
                line2 = fp.readline()
                m = _PYTHON_MAGIC_COMMENT_re.match(line2.decode('ascii', 'ignore'))
        if has_bom:
            if m:
                raise SyntaxError('python refuses to compile code with both a UTF8 byte-order-mark and a magic encoding comment')
            return 'utf_8'
        elif m:
            return m.group(1)
        else:
            return None
    finally:
        fp.seek(pos)