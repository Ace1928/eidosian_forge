from .exceptions import *
from .constants import *
from .dfa import *
from .recognizers import *
from .streams import *
from .tokens import *
def version_str_to_tuple(version_str):
    import re
    import sys
    if version_str == 'HEAD':
        return (sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize)
    m = re.match('(\\d+)\\.(\\d+)(\\.(\\d+))?(b(\\d+))?', version_str)
    if m is None:
        raise ValueError('Bad version string %r' % version_str)
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(4) or 0)
    beta = int(m.group(6) or sys.maxsize)
    return (major, minor, patch, beta)