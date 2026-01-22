import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def get_one_match(expr, lines):
    """
    Must be only one match, otherwise result is None.
    When there is a match, strip leading "[" and trailing "]"
    """
    expr = f'\\[({expr})\\]'
    matches = list(filter(None, (re.search(expr, line) for line in lines)))
    if len(matches) == 1:
        return matches[0].group(1)
    else:
        return None