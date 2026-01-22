import sys, os, subprocess
from .error import PkgConfigError
def get_other_libs(string):
    return [x for x in string.split() if not x.startswith('-L') and (not x.startswith('-l'))]