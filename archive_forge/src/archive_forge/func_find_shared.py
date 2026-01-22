import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def find_shared(paths, name):
    """
    paths is a list of directories to search for an archive.
    name is the abbreviated name given to find_library().
    Process: search "paths" for archive, and if an archive is found
    return the result of get_member().
    If an archive is not found then return None
    """
    for dir in paths:
        if dir == '/lib':
            continue
        base = f'lib{name}.a'
        archive = path.join(dir, base)
        if path.exists(archive):
            members = get_shared(get_ld_headers(archive))
            member = get_member(re.escape(name), members)
            if member is not None:
                return (base, member)
            else:
                return (None, None)
    return (None, None)