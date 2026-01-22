import sys
import os
import re
def find_file(pat, libdir=None):
    if libdir is None:
        libdirs = get_lib_dirs()
    elif isinstance(libdir, str):
        libdirs = [libdir]
    else:
        libdirs = list(libdir)
    files = []
    for ldir in libdirs:
        try:
            entries = os.listdir(ldir)
        except FileNotFoundError:
            continue
        candidates = [os.path.join(ldir, ent) for ent in entries if pat.match(ent)]
        files.extend([c for c in candidates if os.path.isfile(c)])
    return files