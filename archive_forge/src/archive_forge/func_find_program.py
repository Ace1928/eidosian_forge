from __future__ import (absolute_import, division, print_function)
import os
import sys
def find_program(name, executable):
    """
    Find and return the full path to the named program, optionally requiring it to be executable.
    Raises an exception if the program is not found.
    """
    path = os.environ.get('PATH', os.path.defpath)
    seen = set([os.path.abspath(__file__)])
    mode = os.F_OK | os.X_OK if executable else os.F_OK
    for base in path.split(os.path.pathsep):
        candidate = os.path.abspath(os.path.join(base, name))
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate) and os.access(candidate, mode):
            return candidate
    raise Exception('Executable "%s" not found in path: %s' % (name, path))