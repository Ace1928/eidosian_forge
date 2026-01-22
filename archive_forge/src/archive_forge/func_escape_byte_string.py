from __future__ import absolute_import
import re
import sys
def escape_byte_string(s):
    """Escape a byte string so that it can be written into C code.
    Note that this returns a Unicode string instead which, when
    encoded as ISO-8859-1, will result in the correct byte sequence
    being written.
    """
    s = _replace_specials(s)
    try:
        return s.decode('ASCII')
    except UnicodeDecodeError:
        pass
    if IS_PYTHON3:
        s_new = bytearray()
        append, extend = (s_new.append, s_new.extend)
        for b in s:
            if b >= 128:
                extend(('\\%3o' % b).encode('ASCII'))
            else:
                append(b)
        return s_new.decode('ISO-8859-1')
    else:
        l = []
        append = l.append
        for c in s:
            o = ord(c)
            if o >= 128:
                append('\\%3o' % o)
            else:
                append(c)
        return join_bytes(l).decode('ISO-8859-1')