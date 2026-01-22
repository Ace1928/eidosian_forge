import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def globs_to_re(globs):
    """Returns an re object for the given globs.

    Only * and ? wildcards are supported.  Literal * and ? may be matched via
    \\* and \\?, respectively.  A literal backslash is matched \\\\.  Any other
    character after a backslash is forbidden.

    Empty globs match nothing.

    Raises MachineReadableFormatError if any of the globs is illegal.
    """
    buf = io.StringIO()
    for i, glob in enumerate(globs):
        if i != 0:
            buf.write('|')
        i = 0
        n = len(glob)
        while i < n:
            c = glob[i]
            i += 1
            if c == '*':
                buf.write('.*')
            elif c == '?':
                buf.write('.')
            elif c == '\\':
                if i < n:
                    c = glob[i]
                    i += 1
                else:
                    raise MachineReadableFormatError('single backslash not allowed at end')
                if c in '\\?*':
                    buf.write(re.escape(c))
                else:
                    raise MachineReadableFormatError('invalid escape sequence: \\%s' % c)
            else:
                buf.write(re.escape(c))
    buf.write('\\Z')
    return re.compile(buf.getvalue(), re.MULTILINE | re.DOTALL)