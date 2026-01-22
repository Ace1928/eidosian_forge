import warnings
import sys
from urllib import parse as urlparse
from paste.recursive import ForwardRequestException, RecursiveMiddleware, RecursionLoop
from paste.util import converters
from paste.response import replace_header
def error_codes_mapper(code, message, environ, global_conf, codes):
    if code in codes:
        return codes[code]
    else:
        return None