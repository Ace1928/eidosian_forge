import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def removeduppaths():
    """ Remove duplicate entries from sys.path along with making them
    absolute"""
    L = []
    known_paths = set()
    for dir in sys.path:
        dir, dircase = makepath(dir)
        if dircase not in known_paths:
            L.append(dir)
            known_paths.add(dircase)
    sys.path[:] = L
    return known_paths