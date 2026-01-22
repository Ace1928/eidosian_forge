import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def get_legacy(members):
    """
    This routine provides historical aka legacy naming schemes started
    in AIX4 shared library support for library members names.
    e.g., in /usr/lib/libc.a the member name shr.o for 32-bit binary and
    shr_64.o for 64-bit binary.
    """
    if AIX_ABI == 64:
        expr = 'shr4?_?64\\.o'
        member = get_one_match(expr, members)
        if member:
            return member
    else:
        for name in ['shr.o', 'shr4.o']:
            member = get_one_match(re.escape(name), members)
            if member:
                return member
    return None