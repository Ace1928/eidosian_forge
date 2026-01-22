import sys
import types
from cgi import parse_header
def url_unquote(s):
    return unquote(s.encode('ascii')).decode('latin-1')