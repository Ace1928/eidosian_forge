from collections import namedtuple
from . import compat
from . import exceptions
from . import misc
from . import normalizers
from . import uri
def split_authority(authority):
    userinfo = host = port = None
    extra_host = None
    rest = authority
    if '@' in authority:
        userinfo, rest = authority.rsplit('@', 1)
    if rest.startswith('['):
        host, rest = rest.split(']', 1)
        host += ']'
    if ':' in rest:
        extra_host, port = rest.split(':', 1)
    elif not host and rest:
        host = rest
    if extra_host and (not host):
        host = extra_host
    return (userinfo, host, port)