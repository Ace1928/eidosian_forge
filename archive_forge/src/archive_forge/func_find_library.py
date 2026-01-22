import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def find_library(name):
    """AIX implementation of ctypes.util.find_library()
    Find an archive member that will dlopen(). If not available,
    also search for a file (or link) with a .so suffix.

    AIX supports two types of schemes that can be used with dlopen().
    The so-called SystemV Release4 (svr4) format is commonly suffixed
    with .so while the (default) AIX scheme has the library (archive)
    ending with the suffix .a
    As an archive has multiple members (e.g., 32-bit and 64-bit) in one file
    the argument passed to dlopen must include both the library and
    the member names in a single string.

    find_library() looks first for an archive (.a) with a suitable member.
    If no archive+member pair is found, look for a .so file.
    """
    libpaths = get_libpaths()
    base, member = find_shared(libpaths, name)
    if base is not None:
        return f'{base}({member})'
    soname = f'lib{name}.so'
    for dir in libpaths:
        if dir == '/lib':
            continue
        shlib = path.join(dir, soname)
        if path.exists(shlib):
            return soname
    return None