import re
from . import urlutils
from .hooks import Hooks
def parse_cvs_location(location):
    parts = location.split(':')
    if parts[0] or parts[1] not in ('pserver', 'ssh', 'extssh'):
        raise ValueError('not a valid CVS location string')
    try:
        username, hostname = parts[2].split('@', 1)
    except IndexError:
        hostname = parts[2]
        username = None
    scheme = parts[1]
    if scheme == 'extssh':
        scheme = 'ssh'
    try:
        path = parts[3]
    except IndexError:
        raise ValueError('no path element in CVS location %s' % location)
    return (scheme, hostname, username, path)