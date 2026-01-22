import re
from . import compat
from . import misc
def normalize_authority(authority):
    """Normalize an authority tuple to a string."""
    userinfo, host, port = authority
    result = ''
    if userinfo:
        result += normalize_percent_characters(userinfo) + '@'
    if host:
        result += normalize_host(host)
    if port:
        result += ':' + port
    return result