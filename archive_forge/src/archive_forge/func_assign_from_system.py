import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
@StatelessInputTransformer.wrap
def assign_from_system(line):
    """Transform assignment from system commands (e.g. files = !ls)"""
    m = assign_system_re.match(line)
    if m is None:
        return line
    return assign_system_template % m.group('lhs', 'cmd')